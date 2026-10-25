"""Measure the CPU deployment effect of merging a trained LoRA adapter.

Both variants are FP32 and use exactly the same trained adapter. There is no
quantization, retraining or checkpoint selection in this experiment.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import platform
import time

from .optimize import make_schedule, read_jsonl, sha256, summarize_timings, write_json


def merge_schedule(prompts: int, repeats: int, seed: int) -> list[dict]:
    names = {"fp32": "unmerged", "int8": "merged"}
    return [{**row, "order": [names[name] for name in row["order"]]} for row in make_schedule(prompts, repeats, seed)]


def summarize_merge(records: list[dict]) -> dict:
    # Reuse the paired arithmetic, but never expose the helper's precision names.
    aliases = {"unmerged": "fp32", "merged": "int8"}
    if any(row["variant"] not in aliases for row in records):
        raise ValueError("Only unmerged and merged FP32 variants are valid")
    summary = summarize_timings([{**row, "variant": aliases[row["variant"]]} for row in records])
    return {
        "unmerged": summary["fp32"], "merged": summary["int8"],
        "merged_speedup_ratio": summary["int8_speedup_ratio"],
        "paired_speedup_ratios": summary["paired_speedup_ratios"],
        "paired_median_speedup_ratio": summary["paired_median_speedup_ratio"],
        "paired_min_speedup_ratio": summary["paired_min_speedup_ratio"],
        "paired_max_speedup_ratio": summary["paired_max_speedup_ratio"],
    }


def observed_quality_check(before: dict, after: dict, max_logit_difference: float) -> dict:
    checks = {
        "no_json_exact_match_drop": after["generation"]["json_exact_match_rate"] >= before["generation"]["json_exact_match_rate"],
        "no_valid_json_drop": after["generation"]["valid_json_rate"] >= before["generation"]["valid_json_rate"],
        "loss_increase_at_most_1e_minus_5": after["loss"]["loss"] <= before["loss"]["loss"] + 1e-5,
        "logit_max_absolute_difference_at_most_1e_minus_3": max_logit_difference <= 1e-3,
    }
    return {"passed": all(checks.values()), "checks": checks,
            "scope": "Observed checks on this held-out synthetic task and two validation logit probes; not general quality equivalence."}


def run(args: argparse.Namespace) -> dict:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import transformers
    import peft
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    from .training import encode_examples, evaluate_loss, evaluate_generation, render_prompt

    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("A CPU-only PyTorch wheel is required")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    output = Path(args.output)
    recipe_path = output.with_name(output.stem + "-recipe.json")
    if output.exists() or recipe_path.exists():
        raise FileExistsError("Choose a new output path; completed experiment files are not overwritten")
    training_dir = Path(args.training_dir)
    config = json.loads((training_dir / "resolved_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((training_dir / "metrics.json").read_text(encoding="utf-8"))
    adapter_dir = training_dir / "adapter"
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("A completed safetensors adapter is required")
    hashes = {}
    for filename, expected in metrics["artifact_sha256"].items():
        candidate = (adapter_dir / filename).resolve()
        if not candidate.is_relative_to(adapter_dir.resolve()):
            raise ValueError("Invalid adapter artifact path")
        hashes[filename] = sha256(candidate)
        if hashes[filename] != expected:
            raise ValueError(f"Adapter differs from recorded training artifact: {filename}")
    validation_path = training_dir / "data" / "validation.jsonl"
    test_path = training_dir / "data" / "test.jsonl"
    for name, path in (("validation", validation_path), ("test", test_path)):
        if sha256(path) != metrics["dataset"]["sha256"][name]:
            raise ValueError(f"{name} data differs from the recorded training split")
    validation = read_jsonl(validation_path)
    test = read_jsonl(test_path)
    selected_test = test if args.quality_examples == 0 else test[:args.quality_examples]
    options = {"revision": config["revision"], "local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], **options)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(config["model_id"], torch_dtype=torch.float32, attn_implementation="sdpa", **options).cpu()
    unmerged = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False).cpu().float().eval()
    unmerged.config.use_cache = True
    unmerged.generation_config.do_sample = False
    for name in ("temperature", "top_p", "top_k"):
        setattr(unmerged.generation_config, name, None)
    started = time.perf_counter()
    merged = copy.deepcopy(unmerged).merge_and_unload(safe_merge=True).cpu().float().eval()
    merge_seconds = time.perf_counter() - started
    models = {"unmerged": unmerged, "merged": merged}
    devices = sorted({str(tensor.device) for model in models.values() for tensor in list(model.parameters()) + list(model.buffers())})
    if devices != ["cpu"]:
        raise RuntimeError(f"Unexpected tensor placement: {devices}")
    report = {
        "status": "running", "experiment": "FP32 LoRA unmerged versus FP32 adapter merged; no quantization",
        "settings": vars(args), "model_id": config["model_id"], "model_revision": config["revision"],
        "adapter_sha256": hashes,
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "cpu_only": {"torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "tensor_devices": devices, "threads": torch.get_num_threads()},
        "merge_seconds_including_deepcopy": merge_seconds,
        "timing_definition": "Model.generate wall time on pretokenized batch-one inputs; includes prompt processing and generation, excludes tokenization, loading, copying, merging and warmup.",
        "test_examples_evaluated": len(selected_test), "test_examples_available": len(test),
        "dataset_sha256": {"validation": sha256(validation_path), "test": sha256(test_path)},
        "limitations": ["Both model variants remain resident; this is not a per-variant RAM measurement.", "Fixed-token timing differs from naturally stopped quality evaluation.", "Short paired measurements on one machine do not establish statistical significance or broad quality equivalence.", "No checkpoint or training hyperparameters are selected using this test set."],
        "logit_probes": [], "benchmark_records": [], "test": {},
    }
    write_json(output, report)
    timing_examples = validation[:min(args.prompts, len(validation))]
    inputs = [tokenizer(render_prompt(tokenizer, row), return_tensors="pt", add_special_tokens=False) for row in timing_examples]
    with torch.inference_mode():
        for index, values in enumerate(inputs):
            before = unmerged(**values, use_cache=False).logits
            after = merged(**values, use_cache=False).logits
            difference = (before - after).abs()
            report["logit_probes"].append({"prompt_index": index, "max_absolute_difference": float(difference.max()), "mean_absolute_difference": float(difference.mean()), "next_token_argmax_equal": bool(torch.equal(before[:, -1].argmax(-1), after[:, -1].argmax(-1)))})
    write_json(output, report)

    def generate_fixed(model, values, tokens):
        with torch.inference_mode():
            start = time.perf_counter()
            result = model.generate(**values, do_sample=False, min_new_tokens=tokens, max_new_tokens=tokens, use_cache=True, pad_token_id=tokenizer.pad_token_id)
            elapsed = time.perf_counter() - start
        new_ids = result[0, values["input_ids"].shape[1]:]
        if new_ids.numel() != tokens:
            raise RuntimeError("Fixed-length benchmark emitted an unexpected number of tokens")
        return elapsed, tokenizer.decode(new_ids, skip_special_tokens=True)

    for model in models.values():
        for _ in range(args.warmups):
            generate_fixed(model, inputs[0], min(16, args.tokens))
    schedule = merge_schedule(len(inputs), args.repeats, args.seed)
    report["benchmark_schedule"] = schedule
    for pair in schedule:
        for name in pair["order"]:
            elapsed, generated = generate_fixed(models[name], inputs[pair["prompt_index"]], args.tokens)
            report["benchmark_records"].append({
                "pair": pair["pair"], "repeat": pair["repeat"], "prompt_index": pair["prompt_index"], "variant": name,
                "seconds": elapsed, "output_tokens": args.tokens,
                "prompt_tokens": int(inputs[pair["prompt_index"]]["input_ids"].shape[1]),
                "tokens_per_second": args.tokens / elapsed, "output": generated,
            })
            write_json(output, report)
            print(f"pair={pair['pair']} {name}: {args.tokens / elapsed:.2f} tokens/s (including prompt processing)", flush=True)
    report["timing_summary"] = summarize_merge(report["benchmark_records"])
    encoded = encode_examples(tokenizer, selected_test, int(config["max_sequence_length"]))
    for name, model in models.items():
        print(f"Held-out task evaluation: {name}, {len(selected_test)} examples", flush=True)
        report["test"][name] = {
            "loss": evaluate_loss(model, encoded, tokenizer.pad_token_id, batch_size=2),
            "generation": evaluate_generation(model, tokenizer, selected_test, max_new_tokens=args.quality_tokens),
        }
        write_json(output, report)
    maximum = max(row["max_absolute_difference"] for row in report["logit_probes"])
    report["observed_quality_check"] = observed_quality_check(report["test"]["unmerged"], report["test"]["merged"], maximum)
    if report["observed_quality_check"]["passed"]:
        write_json(recipe_path, {
            "format": "cpu-ai-lab-reconstruction-v1", "selected_variant": "fp32",
            "deployment": "FP32 base with trained LoRA adapter merged; no quantization",
            "model_id": config["model_id"], "model_revision": config["revision"],
            "adapter_directory": os.path.relpath(adapter_dir, recipe_path.parent).replace("\\", "/"),
            "adapter_sha256": hashes, "versions": report["versions"], "threads": args.threads,
            "inter_op_threads": 1, "quantization_engine": None,
            "measured_speedup_ratio": report["timing_summary"]["merged_speedup_ratio"],
            "scope": "Task quality checks passed on the recorded test data; speedup may be positive or negative. Inspect the report.",
        })
        report["recipe_file"] = recipe_path.name
    report["status"] = "complete"
    write_json(output, report)
    print(json.dumps({"merged_speedup_ratio": report["timing_summary"]["merged_speedup_ratio"], "observed_quality_pass": report["observed_quality_check"]["passed"], "results": str(output)}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", default="artifacts/train-run")
    parser.add_argument("--output", default="results/merge.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--prompts", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--quality-examples", type=int, default=0, help="0 evaluates every held-out test example")
    parser.add_argument("--quality-tokens", type=int, default=40)
    parser.add_argument("--seed", type=int, default=31415)
    args = parser.parse_args()
    for name in ("threads", "tokens", "prompts", "repeats", "warmups", "quality_tokens"):
        if getattr(args, name) < 1:
            parser.error(f"{name} must be positive")
    if args.quality_examples < 0:
        parser.error("quality-examples must be nonnegative")
    run(args)


if __name__ == "__main__":
    main()
