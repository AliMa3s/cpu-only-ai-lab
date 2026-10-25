"""CPU adaptation for longer passages and explicit missing-answer abstention."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from peft import PeftModel, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import document_v2_data as data
from . import document_v2_metrics as scoring
from .document_task import (directory_hashes, left_pad, loss_evaluation, response_loss,
                            sha256, trim_generated, verify_saved_artifacts)
from .training import (PeakRSS, adapter_snapshot, assert_cpu_model, collate_batch,
                       encode_examples, render_prompt, write_json)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def validate_config(config):
    for key in ("training_threads", "evaluation_threads", "loss_batch_size", "generation_batch_size",
                "generation_tokens", "optimizer_steps", "validation_every", "max_prompt_tokens", "max_sequence_length"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid positive integer: {key}")
    if config["batch_size"] != 4 or type(config["batch_size"]) is not int:
        raise ValueError("Four-stratum training requires global batch four.")
    if isinstance(config["learning_rate"], bool) or not isinstance(config["learning_rate"], (int, float)) or not math.isfinite(config["learning_rate"]) or config["learning_rate"] <= 0:
        raise ValueError("Learning rate must be finite and positive.")
    if config["max_sequence_length"] <= config["max_prompt_tokens"]:
        raise ValueError("Sequence budget must leave room for response targets.")
    return config


def sample_schedule(rows, steps, seed):
    """One distinct example from each stratum per update, cycling only if needed."""
    rng = random.Random(seed)
    groups = {name: [i for i, row in enumerate(rows) if data.stratum(row) == name] for name in data.STRATA}
    if any(not indices for indices in groups.values()):
        raise ValueError("Every training stratum must contain examples.")
    for indices in groups.values():
        rng.shuffle(indices)
    positions = {name: 0 for name in groups}
    result = []
    for _ in range(steps):
        chosen = []
        for name, indices in groups.items():
            if positions[name] == len(indices):
                rng.shuffle(indices)
                positions[name] = 0
            chosen.append(indices[positions[name]])
            positions[name] += 1
        rng.shuffle(chosen)
        result.append(chosen)
    return result


def state_hash(state):
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        digest.update(name.encode())
        digest.update(tensor.contiguous().numpy().tobytes())
    return digest.hexdigest()


def state_change(first, second):
    return math.sqrt(sum(float((first[key] - value).square().sum()) for key, value in second.items()))


def compact(metric):
    return {key: value for key, value in metric.items() if key != "records"}


@torch.inference_mode()
def generation_evaluation(model, tokenizer, rows, batch_size, token_limit):
    model.eval()
    prompts = [tokenizer.encode(render_prompt(tokenizer, row), add_special_tokens=False) for row in rows]
    order = sorted(range(len(rows)), key=lambda i: (len(prompts[i]), i))
    records, local = [None] * len(rows), [None] * len(rows)
    started = time.perf_counter()
    for offset in range(0, len(order), batch_size):
        indices = order[offset:offset + batch_size]
        batch = left_pad([prompts[i] for i in indices], tokenizer.pad_token_id)
        generated = model.generate(**batch, do_sample=False, use_cache=True, max_new_tokens=token_limit,
                                   eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        width = batch["input_ids"].shape[1]
        for position, original in enumerate(indices):
            ids = trim_generated(generated[position, width:].tolist(), tokenizer.eos_token_id)
            answer = tokenizer.decode(ids, skip_special_tokens=True)
            row = rows[original]
            record = {**scoring.score_prediction(row, answer),
                      "prediction_sha256": sha256(answer.encode()),
                      "generated_tokens_including_eos": len(ids),
                      "ended_with_eos": bool(ids and ids[-1] == tokenizer.eos_token_id)}
            records[original] = record
            local[original] = {**record, "question": row["question"], "references": row["references"], "prediction": answer}
        if offset % (batch_size * 4) == 0 or offset + batch_size >= len(rows):
            print(json.dumps({"stage": "generation", "processed": min(offset + batch_size, len(rows)), "total": len(rows)}), flush=True)
    return {**scoring.summarize(records), "seconds": time.perf_counter() - started,
            "generated_tokens_including_eos": sum(r["generated_tokens_including_eos"] for r in records),
            "ended_with_eos": sum(r["ended_with_eos"] for r in records), "records": records}, local


def model_options(model):
    model.generation_config.do_sample = False
    for key in ("temperature", "top_k", "top_p"):
        setattr(model.generation_config, key, None)
    assert_cpu_model(model)


def load_source(config, directory):
    if read(directory / "run_status.json").get("status") != "complete":
        raise ValueError("Warmstart source run must be complete.")
    metrics, previous = read(directory / "metrics.json"), read(directory / "resolved_config.json")
    verify_saved_artifacts(directory, metrics)
    if any(config[key] != previous[key] for key in ("model_id", "revision")):
        raise ValueError("Warmstart model/revision differs from the new recipe.")
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(config["model_id"], revision=config["revision"], local_files_only=True,
                                               trust_remote_code=False, torch_dtype=torch.float32, attn_implementation="sdpa").cpu()
    base_parameters = sum(p.numel() for p in base.parameters())
    model = PeftModel.from_pretrained(base, directory / "adapter", is_trainable=True).cpu()
    model_options(model)
    if sum(p.numel() for p in model.parameters() if p.requires_grad) != metrics["model"]["trainable_parameters"]:
        raise ValueError("Warmstart trainable parameter count changed.")
    evidence = {"metrics_sha256": sha256((directory / "metrics.json").read_bytes()),
                "split_manifest_sha256": metrics["dataset"]["split_manifest_sha256"],
                "adapter_sha256": metrics["adapter_sha256"], "selected_step": metrics["selection"]["step"],
                "all_saved_artifact_checksums_verified": True}
    return model, tokenizer, base_parameters, evidence


def train(config, output, prepared, source):
    validate_config(config)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Use a new output directory; completed and failed runs are preserved.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved_config.json", config)
    write_json(output / "run_status.json", {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()})
    try:
        return _train(config, output, prepared, source)
    except BaseException as error:
        write_json(output / "run_status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
                                               "error_type": type(error).__name__, "error": str(error)[:500]})
        raise


def _train(config, output, prepared, source):
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("A CPU-only PyTorch build is required.")
    torch.set_num_threads(config["training_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    started = time.perf_counter()
    with PeakRSS() as memory:
        splits, provenance = data.load_prepared(prepared, config)
        write_json(output / "data_provenance.json", provenance)
        write_json(output / "split_manifest.json", data.public_manifest(splits))
        model, tokenizer, base_parameters, source_evidence = load_source(config, source)
        if provenance["prior_manifest_sha256"] != source_evidence["split_manifest_sha256"]:
            raise ValueError("Prepared exclusions do not belong to the verified warmstart source.")
        initial = adapter_snapshot(model)
        encoded = {name: encode_examples(tokenizer, rows, config["max_sequence_length"]) for name, rows in splits.items()}
        schedule = sample_schedule(splits["train"], config["optimizer_steps"], config["seed"])
        schedule_ids = [[splits["train"][i]["id"] for i in indices] for indices in schedule]
        write_json(output / "sample_schedule.json", schedule_ids)

        def evaluation(split):
            torch.set_num_threads(config["evaluation_threads"])
            loss = loss_evaluation(model, encoded[split], tokenizer.pad_token_id, config["loss_batch_size"])
            generated, local = generation_evaluation(model, tokenizer, splits[split], config["generation_batch_size"], config["generation_tokens"])
            return {"loss": loss, "generation": generated}, local

        print(json.dumps({"stage": "warmstart_validation"}), flush=True)
        before_validation, local = evaluation("validation")
        write_json(output / "validation-initial-local.json", local)
        print(json.dumps({"stage": "warmstart_test"}), flush=True)
        before_test, local = evaluation("test")
        write_json(output / "test-before-local.json", local)
        before = {"validation": before_validation, "test": before_test}
        write_json(output / "baseline.json", before)
        best_step, best_adapter, best_validation = 0, initial, before_validation
        best_key = None
        trials, curve, seen = [], [], set()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"], weight_decay=0.0)
        compute, input_tokens, target_tokens, padded_tokens = 0.0, 0, 0, 0
        training_started = time.perf_counter()
        for step, indices in enumerate(schedule, 1):
            torch.set_num_threads(config["training_threads"])
            seen.update(indices)
            batch = collate_batch([encoded["train"][i] for i in indices], tokenizer.pad_token_id)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            tick = time.perf_counter()
            loss, count = response_loss(model, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite response training loss.")
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True))
            optimizer.step()
            seconds = time.perf_counter() - tick
            compute += seconds
            target_tokens += count
            input_tokens += int(batch["attention_mask"].sum())
            padded_tokens += batch["input_ids"].numel()
            record = {"step": step, "target_token_loss": float(loss.detach()), "compute_seconds": seconds,
                      "target_tokens": count, "gradient_norm_before_clip": norm}
            if step % config["validation_every"] == 0 or step == config["optimizer_steps"]:
                validation, local = evaluation("validation")
                gate = scoring.checkpoint_gate(before_validation["generation"], validation["generation"])
                key = scoring.checkpoint_key(validation["generation"], validation["loss"]["target_token_loss"])
                snapshot = adapter_snapshot(model)
                trials.append({"step": step, "validation": validation, "gate": gate, "key": list(key), "adapter_state_sha256": state_hash(snapshot)})
                write_json(output / f"validation-step-{step}-local.json", local)
                record["validation"] = {"loss": validation["loss"], "generation": compact(validation["generation"]), "gate": gate}
                if gate["accepted"] and (best_key is None or key > best_key):
                    best_step, best_adapter, best_key, best_validation = step, snapshot, key, validation
                write_json(output / "selection_trials.json", trials)
            curve.append(record)
            write_json(output / "training_curve.json", curve)
            if step % 10 == 0 or step == 1:
                print(json.dumps({"stage": "training", "step": step, "loss": record["target_token_loss"],
                                  "compute_seconds": seconds, "selected_step_so_far": best_step,
                                  "validation": record.get("validation")}), flush=True)
        training_seconds = time.perf_counter() - training_started
        last_adapter = adapter_snapshot(model)
        model.save_pretrained(output / "last_adapter", safe_serialization=True)
        set_peft_model_state_dict(model, best_adapter)
        if state_hash(adapter_snapshot(model)) != state_hash(best_adapter):
            raise RuntimeError("Selected adapter restoration failed.")
        model.save_pretrained(output / "adapter", safe_serialization=True)
        tokenizer.save_pretrained(output / "tokenizer")
        print(json.dumps({"stage": "selected_test", "selected_step": best_step}), flush=True)
        after_test, local = evaluation("test")
        write_json(output / "test-after-local.json", local)
        selected_indices = {index for indices in schedule[:best_step] for index in indices}
        always_abstain = scoring.summarize([scoring.score_prediction(row, "NOT_FOUND") for row in splits["test"]])
        metric = {"format": "cpu-ai-lab-document-v2-v1", "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "model": {"id": config["model_id"], "revision": config["revision"], "base_parameters": base_parameters,
                            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)},
                  "source": source_evidence, "dataset": provenance, "before": before,
                  "after": {"validation": best_validation, "test": after_test}, "always_abstain_test_baseline": always_abstain,
                  "selection": {"selected_step": best_step, "accepted_new_checkpoint": best_step > 0, "test_used_for_selection": False,
                                "criterion": "Among validation-gate-passing checkpoints, maximize harmonic mean of answerable exact match and explicit missing-answer recall, then strict F1, strict EM, then lower validation loss; initial adapter is fallback.",
                                "selected_validation_reuses_checkpoint_evaluation": True,
                                "selected_adapter_state_sha256": state_hash(best_adapter)},
                  "training": {"optimizer_steps": config["optimizer_steps"], "examples_seen": len(schedule) * 4,
                               "unique_examples_seen": len(seen), "selected_checkpoint_examples_seen_this_run": best_step * 4,
                               "selected_checkpoint_unique_examples_seen_this_run": len(selected_indices),
                               "input_tokens": input_tokens, "target_tokens": target_tokens, "padded_token_slots": padded_tokens,
                               "compute_seconds": compute, "loop_seconds_including_validation": training_seconds,
                               "sample_schedule_sha256": sha256((output / "sample_schedule.json").read_bytes()),
                               "initial_adapter_state_sha256": state_hash(initial), "last_adapter_state_sha256": state_hash(last_adapter),
                               "selected_adapter_l2_change": state_change(best_adapter, initial), "last_adapter_l2_change": state_change(last_adapter, initial)},
                  "cpu_evidence": {"torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                                   "parameter_devices": sorted({str(p.device) for p in model.parameters()}),
                                   "optimizer_tensor_devices": sorted({str(v.device) for s in optimizer.state.values() for v in s.values() if isinstance(v, torch.Tensor)}),
                                   "training_threads": config["training_threads"], "evaluation_threads": config["evaluation_threads"], "dtype": "float32"},
                  "environment": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "psutil")},
                  "elapsed_seconds": time.perf_counter() - started, "sampled_peak_process_rss_bytes": memory.peak_bytes,
                  "resolved_config_sha256": sha256((output / "resolved_config.json").read_bytes()),
                  "training_curve_sha256": sha256((output / "training_curve.json").read_bytes()),
                  "selection_trials_sha256": sha256((output / "selection_trials.json").read_bytes()),
                  "data_provenance_sha256": sha256((output / "data_provenance.json").read_bytes()),
                  "adapter_sha256": directory_hashes(output / "adapter"), "last_adapter_sha256": directory_hashes(output / "last_adapter"),
                  "tokenizer_sha256": directory_hashes(output / "tokenizer"),
                  "local_prediction_sha256": {p.name: sha256(p.read_bytes()) for p in output.glob("*-local.json")},
                  "limitations": ["Balanced filtered SQuAD2 subset, not official benchmark scores or production prevalence.",
                                  "Long means 193–512 subword tokens in a complete original passage, not a multipage document or OCR.",
                                  "Prior test IDs and passages are excluded, but official-development article topics were evaluated historically.",
                                  "Historical SQuAD/Wikipedia may have appeared in base-model pretraining; no claim of unseen factual knowledge.",
                                  "Validation gate allows up to 0.05 absolute loss of answerable EM and F1; held-out test is descriptive and never selects settings.",
                                  "Explicit NOT_FOUND is required for strict missing-answer credit; separately reported official normalized scores may credit malformed empty outputs.",
                                  "Raw passages/questions/references/generated text and weights remain in local ignored artifacts.",
                                  "Elapsed time excludes dataset preparation, Python imports/startup and final metrics serialization. It includes warmstart loading, baseline evaluation, training, generated validation, saving and selected test evaluation.",
                                  "Output lengths and workload differ from previous experiments; no new speedup claim is made."]}
        write_json(output / "metrics.json", metric)
        write_json(output / "run_status.json", {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"stage": "complete", "selected_step": best_step, "before": compact(before_test["generation"]),
                          "after": compact(after_test["generation"]), "training_seconds": training_seconds}), flush=True)
        return metric


def load_run(directory, threads):
    if read(directory / "run_status.json").get("status") != "complete":
        raise ValueError("Only completed runs can be loaded.")
    config, metrics = read(directory / "resolved_config.json"), read(directory / "metrics.json")
    verify_saved_artifacts(directory, metrics)
    if sha256((directory / "selection_trials.json").read_bytes()) != metrics["selection_trials_sha256"]:
        raise ValueError("Selection evidence checksum mismatch.")
    if config["model_id"] != metrics["model"]["id"] or config["revision"] != metrics["model"]["revision"]:
        raise ValueError("Saved model/revision mismatch.")
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("CPU-only PyTorch is required.")
    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(config["model_id"], revision=config["revision"], local_files_only=True,
                                               trust_remote_code=False, torch_dtype=torch.float32, attn_implementation="sdpa").cpu()
    model = PeftModel.from_pretrained(base, directory / "adapter", is_trainable=False).eval()
    model_options(model)
    if state_hash(adapter_snapshot(model)) != metrics["selection"]["selected_adapter_state_sha256"]:
        raise ValueError("Loaded adapter state checksum differs.")
    return model, tokenizer, config, metrics


@torch.inference_mode()
def answer(model, tokenizer, config, context, question):
    prompt = data.prompt_text(context, question)
    rendered = render_prompt(tokenizer, {"prompt": prompt})
    inputs = tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
    if len(tokenizer.encode(context, add_special_tokens=False)) > config["max_context_tokens"] or inputs["input_ids"].numel() > config["max_prompt_tokens"]:
        raise ValueError("Complete passage/question exceeds the trained budget; no automatic truncation.")
    output = model.generate(**inputs, do_sample=False, max_new_tokens=config["generation_tokens"], use_cache=True,
                            eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    ids = trim_generated(output[0, inputs["input_ids"].shape[1]:].tolist(), tokenizer.eos_token_id)
    text = tokenizer.decode(ids, skip_special_tokens=True)
    return {"answer": text.strip(), "abstained": text.strip() == "NOT_FOUND", "device": "cpu", "prediction_sha256": sha256(text.encode())}


def replay(directory, prepared, threads):
    model, tokenizer, config, metrics = load_run(directory, threads)
    splits, provenance = data.load_prepared(prepared, config)
    if provenance["split_manifest_sha256"] != metrics["dataset"]["split_manifest_sha256"]:
        raise ValueError("Replay data differs from the experiment.")
    chosen, seen = [], set()
    for row in splits["test"]:
        group = data.stratum(row)
        if group not in seen:
            chosen.append(row)
            seen.add(group)
    expected = {row["id"]: row for row in metrics["after"]["test"]["generation"]["records"]}
    records = []
    for row in chosen:
        result = answer(model, tokenizer, config, row["context"], row["question"])
        records.append({"id": row["id"], "stratum": data.stratum(row), "prediction_sha256": result["prediction_sha256"],
                        "expected_prediction_sha256": expected[row["id"]]["prediction_sha256"],
                        "matches_saved_batched_output": result["prediction_sha256"] == expected[row["id"]]["prediction_sha256"]})
    result = {"selection": "First test record in each of four strata, fixed source order, no selection by output quality",
              "examples": len(records), "matched": sum(r["matches_saved_batched_output"] for r in records),
              "all_artifact_checksums_verified": True, "records": records}
    write_json(directory / "reload_verification.json", result)
    return result


def export(directory, destination, prefix):
    import re
    if not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
        raise ValueError("Invalid publication prefix.")
    if read(directory / "run_status.json").get("status") != "complete":
        raise ValueError("Only completed runs can be exported.")
    metrics = read(directory / "metrics.json")
    verify_saved_artifacts(directory, metrics)
    names = {"metrics.json": "metrics", "resolved_config.json": "config", "training_curve.json": "training-curve",
             "selection_trials.json": "selection-trials", "data_provenance.json": "data-provenance",
             "split_manifest.json": "split-manifest", "sample_schedule.json": "sample-schedule"}
    for filename, field in (("selection_trials.json", metrics["selection_trials_sha256"]),
                            ("sample_schedule.json", metrics["training"]["sample_schedule_sha256"])):
        if sha256((directory / filename).read_bytes()) != field:
            raise ValueError("Publication evidence checksum mismatch.")
    if (directory / "reload_verification.json").exists():
        names["reload_verification.json"] = "reload"
    copies = [(directory / name, destination / f"{prefix}-{suffix}.json") for name, suffix in names.items()]
    for source, target in copies:
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise FileExistsError("Existing published evidence has different bytes.")
    destination.mkdir(parents=True, exist_ok=True)
    for source, target in copies:
        target.write_bytes(source.read_bytes())
    return {target.name: sha256(target.read_bytes()) for _, target in copies}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/document-v2.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepared-data", type=Path, default=Path("artifacts/document-v2-data"))
    parser.add_argument("--source-run", type=Path, default=Path("artifacts/document-task-v1"))
    parser.add_argument("--load", type=Path)
    parser.add_argument("--context")
    parser.add_argument("--question")
    parser.add_argument("--threads", type=int, default=12)
    parser.add_argument("--verify-reload", type=Path)
    parser.add_argument("--export-run", type=Path)
    parser.add_argument("--results-directory", type=Path, default=Path("results"))
    parser.add_argument("--export-prefix", default="document-v2")
    args = parser.parse_args()
    if args.export_run:
        print(json.dumps(export(args.export_run, args.results_directory, args.export_prefix)))
    elif args.verify_reload:
        print(json.dumps(replay(args.verify_reload, args.prepared_data, args.threads)))
    elif args.load:
        if not args.context or not args.question:
            parser.error("--load requires --context and --question")
        model, tokenizer, config, _ = load_run(args.load, args.threads)
        print(json.dumps(answer(model, tokenizer, config, args.context, args.question)))
    else:
        if args.output is None:
            parser.error("Training requires --output")
        train(read(args.config), args.output, args.prepared_data, args.source_run)


if __name__ == "__main__":
    main()
