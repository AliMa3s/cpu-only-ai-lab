"""CPU-only LoRA task adaptation with auditable held-out measurements.

Run with ``python -m cpu_ai_lab.training --config configs/training.json
--output artifacts/train-run``. All tensors are explicitly on CPU, the default
configuration requires a CPU-only PyTorch build, and no remote code is executed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import random
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import psutil
import torch
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from .data import load_splits, make_extraction_splits, save_splits, score_response


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def assert_cpu_model(model) -> None:
    devices = {str(value.device) for value in list(model.parameters()) + list(model.buffers())}
    if devices - {"cpu"}:
        raise RuntimeError(f"Non-CPU tensor detected: {sorted(devices)}")


def render_prompt(tokenizer, example: dict) -> str:
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": example["prompt"]}],
            tokenize=False,
            add_generation_prompt=True,
        )
    return "User: " + example["prompt"] + "\nAssistant: "


def encode_examples(tokenizer, examples: list[dict], max_length: int) -> list[dict]:
    encoded = []
    if tokenizer.eos_token_id is None:
        raise ValueError("A tokenizer EOS token is required.")
    for example in examples:
        prompt_ids = tokenizer.encode(render_prompt(tokenizer, example), add_special_tokens=False)
        target_ids = tokenizer.encode(example["response"], add_special_tokens=False) + [tokenizer.eos_token_id]
        ids = prompt_ids + target_ids
        if len(ids) > max_length:
            raise ValueError(f"Example {example.get('id', '')} has {len(ids)} tokens; maximum is {max_length}. Refusing silent truncation.")
        encoded.append({"input_ids": ids, "labels": [-100] * len(prompt_ids) + target_ids})
    return encoded


def collate_batch(examples: list[dict], pad_token_id: int) -> dict[str, torch.Tensor]:
    width = max(len(example["input_ids"]) for example in examples)
    input_ids, attention_mask, labels = [], [], []
    for example in examples:
        padding = width - len(example["input_ids"])
        input_ids.append(example["input_ids"] + [pad_token_id] * padding)
        attention_mask.append([1] * len(example["input_ids"]) + [0] * padding)
        labels.append(example["labels"] + [-100] * padding)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long, device="cpu"),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long, device="cpu"),
        "labels": torch.tensor(labels, dtype=torch.long, device="cpu"),
    }


@torch.inference_mode()
def evaluate_loss(model, encoded: list[dict], pad_token_id: int, batch_size: int = 2) -> dict:
    if not encoded or batch_size < 1:
        raise ValueError("Evaluation requires nonempty data and a positive batch size.")
    model.eval()
    total_nll, total_tokens = 0.0, 0
    started = time.perf_counter()
    for offset in range(0, len(encoded), batch_size):
        batch = collate_batch(encoded[offset:offset + batch_size], pad_token_id)
        count = int((batch["labels"][:, 1:] != -100).sum())
        loss = float(model(**batch, use_cache=False).loss)
        if not math.isfinite(loss):
            raise RuntimeError("Non-finite evaluation loss.")
        total_nll += loss * count
        total_tokens += count
    average = total_nll / total_tokens
    return {
        "loss": average,
        "target_tokens": total_tokens,
        "perplexity": math.exp(average) if average < 700 else None,
        "seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def evaluate_generation(model, tokenizer, examples: list[dict], max_new_tokens: int = 40, sample_limit: int | None = None) -> dict:
    model.eval()
    selected = examples if sample_limit is None else examples[:sample_limit]
    if not selected:
        raise ValueError("Generation evaluation requires at least one example.")
    started = time.perf_counter()
    records = []
    total_output_tokens = 0
    for example in selected:
        prompt = render_prompt(tokenizer, example)
        inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
        inputs = {name: tensor.to("cpu") for name, tensor in inputs.items()}
        run_started = time.perf_counter()
        generated = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            use_cache=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
        seconds = time.perf_counter() - run_started
        new_ids = generated[0, inputs["input_ids"].shape[1]:]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        score = score_response(text, example["response"])
        total_output_tokens += int(new_ids.numel())
        records.append({
            "id": example.get("id"),
            "prompt": example["prompt"],
            "expected": example["response"],
            "generated": text,
            "generated_tokens_including_eos": int(new_ids.numel()),
            "generation_seconds_including_prefill": seconds,
            **score,
        })
    elapsed = time.perf_counter() - started
    return {
        "examples": len(records),
        "json_exact_match_rate": sum(record["json_exact"] for record in records) / len(records),
        "valid_json_rate": sum(record["valid_json"] for record in records) / len(records),
        "text_exact_match_rate": sum(record["text_exact"] for record in records) / len(records),
        "generated_tokens_including_eos": total_output_tokens,
        "seconds": elapsed,
        "end_to_end_generated_tokens_per_second": total_output_tokens / elapsed,
        "records": records,
    }


class PeakRSS:
    """Sample this process's resident memory; not a system-wide or exact peak."""
    def __init__(self):
        self.peak_bytes = 0
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        process = psutil.Process()
        while not self.stop_event.is_set():
            self.peak_bytes = max(self.peak_bytes, process.memory_info().rss)
            self.stop_event.wait(0.1)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()


def adapter_snapshot(model) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in get_peft_model_state_dict(model).items()}


def _run(config: dict, output: Path, data_directory: Path | None = None) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is not empty: {output}. Use a new run directory.")
    output.mkdir(parents=True, exist_ok=True)
    for key in ("optimizer_steps", "threads", "batch_size", "gradient_accumulation_steps", "validation_every", "generation_samples", "generation_max_new_tokens"):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive.")
    write_json(output / "resolved_config.json", config)
    write_json(output / "run_status.json", {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()})
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if config.get("require_cpu_build", True) and (torch.version.cuda is not None or torch.cuda.is_available()):
        raise RuntimeError("This experiment requires a CPU-only PyTorch build. Install torch from the CPU wheel index.")
    overall_started = time.perf_counter()
    with PeakRSS() as memory:
        splits = load_splits(data_directory) if data_directory else make_extraction_splits(
            config["train_size"], config["validation_size"], config["test_size"], config["seed"]
        )
        data_hashes = save_splits(splits, output / "data")
        print(json.dumps({"stage": "loading_model", "model": config["model_id"]}), flush=True)
        load_options = {"revision": config["revision"], "local_files_only": config.get("local_files_only", False), "trust_remote_code": False}
        tokenizer = AutoTokenizer.from_pretrained(config["model_id"], **load_options)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            config["model_id"], torch_dtype=torch.float32, attn_implementation="sdpa", **load_options
        ).to("cpu")
        base.config.use_cache = False
        base_parameter_count = sum(parameter.numel() for parameter in base.parameters())
        model = get_peft_model(base, LoraConfig(
            task_type="CAUSAL_LM",
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["lora_target_modules"],
            bias="none",
        ))
        assert_cpu_model(model)
        initial_adapter = adapter_snapshot(model)
        encoded = {name: encode_examples(tokenizer, rows, config["max_sequence_length"]) for name, rows in splits.items()}
        print(json.dumps({"stage": "baseline_evaluation", "base_parameters": base_parameter_count}), flush=True)
        before_validation = evaluate_loss(model, encoded["validation"], tokenizer.pad_token_id, config["batch_size"])
        before_test_loss = evaluate_loss(model, encoded["test"], tokenizer.pad_token_id, config["batch_size"])
        before_test_generation = evaluate_generation(model, tokenizer, splits["test"], config["generation_max_new_tokens"], config["generation_samples"])
        write_json(output / "baseline.json", {"validation": before_validation, "test_loss": before_test_loss, "test_generation": before_test_generation})
        # Held-out test metrics are recorded, but never used in this selection.
        best_loss = before_validation["loss"]
        best_step = 0
        best_adapter = initial_adapter
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=config["learning_rate"], weight_decay=config["weight_decay"],
        )
        rng = random.Random(config["seed"])
        order = list(range(len(encoded["train"])))
        rng.shuffle(order)
        position = 0
        curve = []
        examples_seen = 0
        training_target_tokens = 0
        training_input_tokens = 0
        training_compute_seconds = 0.0
        train_started = time.perf_counter()
        for step in range(1, config["optimizer_steps"] + 1):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            step_started = time.perf_counter()
            step_nll, step_tokens = 0.0, 0
            for _ in range(config["gradient_accumulation_steps"]):
                indices = []
                for _ in range(config["batch_size"]):
                    if position >= len(order):
                        rng.shuffle(order)
                        position = 0
                    indices.append(order[position])
                    position += 1
                batch = collate_batch([encoded["train"][index] for index in indices], tokenizer.pad_token_id)
                loss = model(**batch, use_cache=False).loss
                if not torch.isfinite(loss):
                    raise RuntimeError(f"Non-finite training loss at step {step}.")
                (loss / config["gradient_accumulation_steps"]).backward()
                count = int((batch["labels"][:, 1:] != -100).sum())
                step_nll += float(loss.detach()) * count
                step_tokens += count
                examples_seen += len(indices)
                training_input_tokens += int(batch["attention_mask"].sum())
            torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], config["gradient_clip"], error_if_nonfinite=True)
            optimizer.step()
            step_seconds = time.perf_counter() - step_started
            training_compute_seconds += step_seconds
            training_target_tokens += step_tokens
            record = {"step": step, "training_loss": step_nll / step_tokens, "step_seconds": step_seconds}
            if step % config["validation_every"] == 0 or step == config["optimizer_steps"]:
                validation = evaluate_loss(model, encoded["validation"], tokenizer.pad_token_id, config["batch_size"])
                record["validation_loss"] = validation["loss"]
                if validation["loss"] < best_loss:
                    best_loss = validation["loss"]
                    best_step = step
                    best_adapter = adapter_snapshot(model)
            curve.append(record)
            write_json(output / "training_curve.json", curve)
            print(json.dumps({"stage": "training", **record}), flush=True)
        training_wall_seconds = time.perf_counter() - train_started
        final_adapter = adapter_snapshot(model)
        set_peft_model_state_dict(model, best_adapter)
        restored_adapter = get_peft_model_state_dict(model)
        if restored_adapter.keys() != best_adapter.keys() or any(
            not torch.equal(restored_adapter[name].detach().cpu(), tensor)
            for name, tensor in best_adapter.items()
        ):
            raise RuntimeError("The restored adapter does not match the selected validation checkpoint.")
        assert_cpu_model(model)
        model.eval()
        model.save_pretrained(output / "adapter", safe_serialization=True)
        tokenizer.save_pretrained(output / "tokenizer")
        print(json.dumps({"stage": "final_evaluation", "selected_step": best_step}), flush=True)
        after_validation = evaluate_loss(model, encoded["validation"], tokenizer.pad_token_id, config["batch_size"])
        after_test_loss = evaluate_loss(model, encoded["test"], tokenizer.pad_token_id, config["batch_size"])
        after_test_generation = evaluate_generation(model, tokenizer, splits["test"], config["generation_max_new_tokens"], config["generation_samples"])
        final_delta = math.sqrt(sum(float((final_adapter[name] - tensor).square().sum()) for name, tensor in initial_adapter.items()))
        selected_delta = math.sqrt(sum(float((best_adapter[name] - tensor).square().sum()) for name, tensor in initial_adapter.items()))
        metrics = {
            "experiment": "CPU-only LoRA on a controlled synthetic JSON field-extraction task",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "model": {"id": config["model_id"], "revision": config["revision"], "resolved_revision": getattr(base.config, "_commit_hash", None), "base_parameters": base_parameter_count, "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)},
            "cpu_evidence": {"torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "parameter_devices": sorted({str(parameter.device) for parameter in model.parameters()}), "buffer_devices": sorted({str(buffer.device) for buffer in model.buffers()}), "dtype": "float32", "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()},
            "environment": {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(), "processor": platform.processor(), "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "psutil")}},
            "dataset": {"source": "explicit local JSONL files" if data_directory else "deterministic synthetic data; disjoint held-out field values", "counts": {name: len(rows) for name, rows in splits.items()}, "sha256": data_hashes},
            "selection": {"criterion": "lowest validation target-token loss, including untrained baseline", "selected_step": best_step, "test_used_for_selection": False},
            "before": {"validation_loss": before_validation, "test_loss": before_test_loss, "test_generation": before_test_generation},
            "after": {"validation_loss": after_validation, "test_loss": after_test_loss, "test_generation": after_test_generation},
            "training": {"optimizer_steps": config["optimizer_steps"], "examples_seen": examples_seen, "target_tokens": training_target_tokens, "input_tokens_including_prompts": training_input_tokens, "compute_seconds_excluding_validation": training_compute_seconds, "wall_seconds_including_validation": training_wall_seconds, "input_tokens_per_compute_second": training_input_tokens / training_compute_seconds, "final_adapter_l2_change": final_delta, "selected_adapter_l2_change": selected_delta},
            "memory": {"sampled_peak_process_rss_bytes": memory.peak_bytes, "sampling_interval_seconds": 0.1},
            "elapsed_seconds": time.perf_counter() - overall_started,
            "artifact_sha256": {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (output / "adapter").glob("*") if path.is_file()},
            "limitations": ["This is a narrow synthetic adaptation experiment, not general LLM quality evaluation.", "The pretrained base model's original training is not claimed to have been CPU-only.", "Test metrics are reported before/after; only validation loss selects the checkpoint.", "Loss includes response and EOS tokens only and is weighted by the shifted, unmasked target-token count.", "Generation throughput includes prompt processing and Python overhead and is not a pure decoding benchmark.", "RSS is sampled process memory, not an exact peak or total machine memory.", "Results can vary across software versions and CPU execution despite fixed random seeds."],
        }
        if config.get("save_merged_model", False):
            model.merge_and_unload().save_pretrained(output / "merged", safe_serialization=True)
            tokenizer.save_pretrained(output / "merged")
        write_json(output / "metrics.json", metrics)
        write_json(output / "run_status.json", {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"stage": "complete", "output": str(output), "before_test_loss": before_test_loss["loss"], "after_test_loss": after_test_loss["loss"], "before_json_exact": before_test_generation["json_exact_match_rate"], "after_json_exact": after_test_generation["json_exact_match_rate"], "training_seconds": training_wall_seconds}), flush=True)
        return metrics


def run(config: dict, output: Path, data_directory: Path | None = None) -> dict:
    # Never modify the status of an older run when refusing to overwrite it.
    owns_output = not output.exists() or not any(output.iterdir())
    try:
        return _run(config, output, data_directory)
    except BaseException as error:
        if owns_output and output.exists():
            write_json(output / "run_status.json", {
                "status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                "ended_utc": datetime.now(timezone.utc).isoformat(),
                "error_type": type(error).__name__,
                "error": str(error)[:1000],
            })
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/training.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--generation-samples", type=int)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--save-merged", action="store_true")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for argument, key in ((args.steps, "optimizer_steps"), (args.threads, "threads"), (args.generation_samples, "generation_samples")):
        if argument is not None:
            config[key] = argument
    if args.local_files_only:
        config["local_files_only"] = True
    if args.save_merged:
        config["save_merged_model"] = True
    run(config, args.output, args.data_dir)


if __name__ == "__main__":
    main()
