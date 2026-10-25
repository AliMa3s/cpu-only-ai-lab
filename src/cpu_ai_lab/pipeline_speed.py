"""Controlled CPU thread, batching and padding experiments for SMS adaptation.

Tuning and confirmation use separate records from the saved TRAIN split only.
Training keeps each global batch and update intact; microbatch losses are
weighted by their example counts. No tokens, examples, or evaluations are cut.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from peft import PeftModel, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from .real_task import classification_metrics, collate_classification, encode_split, label_logits
from .training import PeakRSS, adapter_snapshot, assert_cpu_model


def write_report(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def batches_by_length(encoded: list[dict], batch_size: int, sort_by_length: bool) -> list[list[int]]:
    if batch_size < 1 or not encoded:
        raise ValueError("Nonempty examples and a positive batch size are required.")
    indices = list(range(len(encoded)))
    if sort_by_length:
        indices.sort(key=lambda index: (len(encoded[index]["input_ids"]), index))
    return [indices[offset:offset + batch_size] for offset in range(0, len(indices), batch_size)]


@torch.inference_mode()
def optimized_evaluate(model, encoded, pad_token_id, token_ids, batch_size,
                       sort_by_length=True, threads=None, capture_logits=False):
    """Same scoring as real_task.evaluate; restore the original record order."""
    if threads is not None:
        torch.set_num_threads(threads)
    model.eval()
    started = time.perf_counter()
    records = [None] * len(encoded)
    true_tokens, padded_tokens, batch_count = 0, 0, 0
    for indices in batches_by_length(encoded, batch_size, sort_by_length):
        rows = [encoded[index] for index in indices]
        batch, labels = collate_classification(rows, pad_token_id)
        logits = label_logits(model, batch, token_ids)
        if not torch.isfinite(logits).all():
            raise RuntimeError("Nonfinite classification logits.")
        nll = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        probabilities = logits.softmax(-1)[:, 1]
        predictions = logits.argmax(-1)
        true_tokens += int(batch["attention_mask"].sum())
        padded_tokens += int(batch["input_ids"].numel())
        batch_count += 1
        for index, row, truth, pred, probability, loss, raw in zip(
                indices, rows, labels.tolist(), predictions.tolist(), probabilities.tolist(), nll.tolist(), logits.tolist()):
            record = {"id": row["id"], "text_sha256": row["text_sha256"], "expected": truth,
                      "predicted": pred, "spam_probability_conditioned_on_two_labels": probability,
                      "binary_label_nll": loss, "input_tokens": len(row["input_ids"]),
                      "message_truncated": row["truncated"]}
            if capture_logits:
                record["label_logits"] = raw
            records[index] = record
    return {**classification_metrics([row["expected"] for row in records], [row["predicted"] for row in records]),
            "binary_label_loss": sum(row["binary_label_nll"] for row in records) / len(records),
            "seconds": time.perf_counter() - started, "records": records,
            "execution": {"threads": torch.get_num_threads(), "batch_size": batch_size,
                          "sort_by_length": sort_by_length, "batches": batch_count,
                          "input_tokens": true_tokens, "processed_tokens_including_padding": padded_tokens,
                          "padding_tokens": padded_tokens - true_tokens}}


def optimized_train_step(model, optimizer, rows, pad_token_id, token_ids,
                         microbatch_size=None, threads=None, gradient_clip=1.0):
    """One optimizer update over unchanged global rows, with weighted gradients."""
    if not rows or (microbatch_size is not None and microbatch_size < 1):
        raise ValueError("Nonempty global batch and positive microbatch size required.")
    if threads is not None:
        torch.set_num_threads(threads)
    microbatch_size = microbatch_size or len(rows)
    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    weighted_loss, true_tokens, padded_tokens = 0.0, 0, 0
    for offset in range(0, len(rows), microbatch_size):
        micro = rows[offset:offset + microbatch_size]
        batch, targets = collate_classification(micro, pad_token_id)
        logits = label_logits(model, batch, token_ids)
        loss = torch.nn.functional.cross_entropy(logits, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite training loss.")
        weight = len(micro) / len(rows)
        (loss * weight).backward()
        weighted_loss += float(loss.detach()) * weight
        true_tokens += int(batch["attention_mask"].sum())
        padded_tokens += int(batch["input_ids"].numel())
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    norm = float(torch.nn.utils.clip_grad_norm_(parameters, gradient_clip, error_if_nonfinite=True))
    optimizer.step()
    return {"binary_label_loss": weighted_loss, "seconds": time.perf_counter() - started,
            "global_batch_size": len(rows), "microbatch_size": microbatch_size,
            "input_tokens": true_tokens, "processed_tokens_including_padding": padded_tokens,
            "padding_tokens": padded_tokens - true_tokens, "gradient_norm_before_clip": norm,
            "threads": torch.get_num_threads()}


def compare_predictions(reference: dict, candidate: dict, tolerance=1e-4) -> dict:
    first, second = reference["records"], candidate["records"]
    if [(r["id"], r["text_sha256"], r["expected"], r["input_tokens"]) for r in first] != [(r["id"], r["text_sha256"], r["expected"], r["input_tokens"]) for r in second]:
        raise ValueError("Evaluation rows, labels, order or input token counts changed.")
    max_probability = max(abs(a["spam_probability_conditioned_on_two_labels"] - b["spam_probability_conditioned_on_two_labels"]) for a, b in zip(first, second))
    max_nll = max(abs(a["binary_label_nll"] - b["binary_label_nll"]) for a, b in zip(first, second))
    has_logits = all(len(row.get("label_logits", [])) == 2 for row in first + second)
    max_logit = max(abs(x - y) for a, b in zip(first, second) for x, y in zip(a["label_logits"], b["label_logits"])) if has_logits else None
    disagreements = sum(a["predicted"] != b["predicted"] for a, b in zip(first, second))
    return {"prediction_disagreements": disagreements, "max_abs_probability_difference": max_probability,
            "max_abs_nll_difference": max_nll, "max_abs_logit_difference": max_logit,
            "logits_compared": has_logits, "absolute_tolerance": tolerance,
            "accepted": disagreements == 0 and max_probability <= tolerance and max_nll <= tolerance and (max_logit is None or max_logit <= tolerance)}


def validate_config(config):
    for name in ("baseline_threads", "baseline_eval_batch_size", "tuning_examples", "confirmation_examples", "tuning_train_steps", "confirmation_train_steps"):
        if type(config.get(name)) is not int or config[name] < 1:
            raise ValueError(f"{name} must be a positive integer.")
    if type(config.get("confirmation_repeats")) is not int or config["confirmation_repeats"] < 3:
        raise ValueError("At least three confirmation repeats are required.")
    for name in ("thread_candidates", "eval_batch_candidates", "microbatch_candidates"):
        entries = config.get(name)
        if not isinstance(entries, list) or not entries or len(set(entries)) != len(entries) or any(type(x) is not int or x < 1 for x in entries):
            raise ValueError(f"{name} must contain distinct positive integers.")
    if config["baseline_threads"] not in config["thread_candidates"] or 4 not in config["microbatch_candidates"]:
        raise ValueError("The search must include the baseline settings.")
    if any(x > 4 for x in config["microbatch_candidates"]):
        raise ValueError("Microbatches cannot exceed the fixed global batch of four.")
    if config["tuning_examples"] < config["tuning_train_steps"] * 4 or config["confirmation_examples"] <= config["confirmation_train_steps"] * 4:
        raise ValueError("Need enough full training batches and separate confirmation score records.")
    for name in ("logit_tolerance", "adapter_tolerance", "training_logit_tolerance"):
        if not isinstance(config.get(name), (int, float)) or not math.isfinite(config[name]) or config[name] <= 0:
            raise ValueError("Tolerances must be finite and positive.")
    return config


def paired_statistics(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("No measured pairs.")
    ratios = [r["baseline_seconds"] / r["candidate_seconds"] for r in rows]
    if any(not math.isfinite(x) or x <= 0 for x in ratios):
        raise ValueError("Invalid paired timings.")
    return {"pairs": len(rows), "median_paired_speedup": statistics.median(ratios),
            "aggregate_speedup": sum(r["baseline_seconds"] for r in rows) / sum(r["candidate_seconds"] for r in rows),
            "min_speedup": min(ratios), "max_speedup": max(ratios), "faster_pairs": sum(r > 1 for r in ratios),
            "all_quality_checks_pass": all(r["quality"]["accepted"] for r in rows)}


def load_training_source(directory: Path):
    status = json.loads((directory / "run_status.json").read_text())
    if status.get("status") != "complete":
        raise ValueError("Only a completed source training run can be used.")
    cfg = json.loads((directory / "resolved_config.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    for name, expected in metrics["adapter_sha256"].items():
        if Path(name).name != name or sha256(directory / "adapter" / name) != expected:
            raise ValueError("Source adapter checksum mismatch.")
    rows = [json.loads(line) for line in (directory / "data" / "train.jsonl").read_text(encoding="utf-8").splitlines()]
    manifest = json.loads((directory / "split_manifest.json").read_text())
    if sha256(directory / "split_manifest.json") != metrics["dataset"]["split_manifest_sha256"]:
        raise ValueError("Source split manifest changed.")
    if [{k: v for k, v in row.items() if k != "text"} for row in rows] != manifest["train"]:
        raise ValueError("Training records do not match the source manifest.")
    if any(hashlib.sha256(row["text"].encode()).hexdigest() != row["text_sha256"] for row in rows):
        raise ValueError("Training message hash mismatch.")
    return cfg, metrics, rows


def run(args):
    if args.output.exists():
        raise FileExistsError("Choose a new report path.")
    config = validate_config(json.loads(args.config.read_text()))
    source_cfg, source_metrics, rows = load_training_source(args.training_dir)
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("CPU-only PyTorch is required.")
    torch.set_num_interop_threads(1)
    torch.set_num_threads(config["baseline_threads"])
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["seed"])
    rng = random.Random(config["seed"])
    order = list(range(len(rows)))
    rng.shuffle(order)
    tuning_rows = [rows[i] for i in order[:config["tuning_examples"]]]
    confirmation_rows = [rows[i] for i in order[config["tuning_examples"]:config["tuning_examples"] + config["confirmation_examples"]]]
    if len(confirmation_rows) != config["confirmation_examples"]:
        raise ValueError("Insufficient separate training-only benchmark examples.")
    report = {"format": "cpu-ai-lab-pipeline-speed-v1", "status": "running",
              "started_utc": datetime.now(timezone.utc).isoformat(), "config": config,
              "source": {"model_id": source_cfg["model_id"], "revision": source_cfg["revision"],
                         "configuration": source_cfg, "config_sha256": sha256(args.training_dir / "resolved_config.json"),
                         "metrics_sha256": sha256(args.training_dir / "metrics.json"),
                         "adapter_sha256": source_metrics["adapter_sha256"],
                         "split_manifest_sha256": source_metrics["dataset"]["split_manifest_sha256"]},
              "data": {"selection_source": "Saved TRAIN split only; validation/test never used",
                       "tuning_ids": [r["id"] for r in tuning_rows], "confirmation_ids": [r["id"] for r in confirmation_rows]},
              "method": {"search": "Hierarchical: evaluation threads then batch/order; training threads then microbatch size. This is not an exhaustive joint search.",
                         "timing": "Eval includes collation, forward and scoring; train includes collation, forward/backward/clip/AdamW and work counters. The historical real_task compute timer excluded collation. Thread switching, adapter restoration and fresh optimizer creation are outside microbenchmark timing; whole-run follow-up includes them. Use whole-run wall time for the final speed claim.",
                         "training_work": "Same ordered four-example global batches and one update each. Microbatch mean losses weighted by examples/global size. Identical saved adapter and fresh AdamW restored before each trajectory.",
                         "precision": "FP32 only; SDPA; no truncation changes, quantization, skipped examples or reduced validation.",
                         "selection": "Tuning only; require identical eval predictions and small logit differences. Training allows declared adapter-update and confirmation-logit tolerances; whole-run quality must be checked separately.",
                         "memory": "Sampled process RSS; same process can retain allocator memory across variants, so per-variant values are not isolated memory requirements."},
              "warmups": [], "tuning": [], "confirmation": {"evaluation": [], "training": []}}
    write_report(args.output, report)
    try:
        with PeakRSS() as memory:
            opts = {"revision": source_cfg["revision"], "local_files_only": True, "trust_remote_code": False}
            tokenizer = AutoTokenizer.from_pretrained(source_cfg["model_id"], **opts)
            if tokenizer.pad_token_id is None:
                tokenizer.pad_token = tokenizer.eos_token
            labels = [tokenizer.encode(s, add_special_tokens=False) for s in ("0", "1")]
            if any(len(value) != 1 for value in labels) or labels[0] == labels[1]:
                raise ValueError("The two classification labels must be distinct single tokens.")
            token_ids = [value[0] for value in labels]
            base = AutoModelForCausalLM.from_pretrained(source_cfg["model_id"], torch_dtype=torch.float32, attn_implementation="sdpa", **opts).cpu()
            model = PeftModel.from_pretrained(base, args.training_dir / "adapter", is_trainable=True).cpu()
            assert_cpu_model(model)
            if sum(p.numel() for p in model.parameters() if p.requires_grad) != source_metrics["model"]["trainable_parameters"]:
                raise ValueError("Trainable parameter count differs from the source adaptation run.")
            if getattr(base.config, "attention_dropout", 0.0) or any(isinstance(m, torch.nn.Dropout) and m.p > 0 for m in model.modules()):
                raise ValueError("Fixed-work microbatch equivalence currently requires zero training dropout.")
            initial = adapter_snapshot(model)
            tune = encode_split(tokenizer, tuning_rows, source_cfg["max_sequence_length"])
            confirm = encode_split(tokenizer, confirmation_rows, source_cfg["max_sequence_length"])
            pad = tokenizer.pad_token_id

            def evaluation(examples, setting):
                return optimized_evaluate(model, examples, pad, token_ids, capture_logits=True, **setting)

            def training(examples, setting, steps):
                set_peft_model_state_dict(model, initial)
                restored = adapter_snapshot(model)
                if restored.keys() != initial.keys() or any(not torch.equal(restored[name], value) for name, value in initial.items()):
                    raise RuntimeError("Training benchmark adapter restoration failed.")
                torch.manual_seed(config["seed"])
                torch.set_num_threads(setting["threads"])
                optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=source_cfg["learning_rate"], weight_decay=0.0)
                history = []
                for step in range(steps):
                    batch = examples[step * 4:(step + 1) * 4]
                    if len(batch) != 4:
                        raise ValueError("Insufficient full global batches.")
                    history.append(optimized_train_step(model, optimizer, batch, pad, token_ids, microbatch_size=setting["microbatch_size"]))
                state = adapter_snapshot(model)
                digest = hashlib.sha256()
                for name, tensor in sorted(state.items()):
                    digest.update(name.encode()); digest.update(tensor.contiguous().numpy().tobytes())
                result = {"seconds": sum(r["seconds"] for r in history), "steps": history,
                          "input_tokens": sum(r["input_tokens"] for r in history),
                          "processed_tokens_including_padding": sum(r["processed_tokens_including_padding"] for r in history),
                          "updated_adapter_sha256": digest.hexdigest(),
                          "optimizer_steps_per_parameter": sorted({int(v["step"]) for v in optimizer.state.values()}),
                          "optimizer_tensor_devices": sorted({str(t.device) for v in optimizer.state.values() for t in v.values() if isinstance(t, torch.Tensor)})}
                return result, state

            def train_agreement(a, b):
                differences = [float((a[key] - b[key]).abs().max()) for key in a]
                delta = max(differences)
                return {"max_abs_adapter_difference": delta, "tolerance": config["adapter_tolerance"], "accepted": delta <= config["adapter_tolerance"]}

            eval_baseline_setting = {"threads": config["baseline_threads"], "batch_size": config["baseline_eval_batch_size"], "sort_by_length": False}
            train_baseline_setting = {"threads": config["baseline_threads"], "microbatch_size": 4}
            # Warmups are not used for settings selection or timing summaries.
            for threads in config["thread_candidates"]:
                result = evaluation(tune[:8], {**eval_baseline_setting, "threads": threads})
                report["warmups"].append({"stage": "evaluation", "threads": threads,
                                          "examples": len(tune[:8]), "seconds": result["seconds"], "execution": result["execution"]})
            eval_reference = evaluation(tune, eval_baseline_setting)
            train_reference, train_reference_state = training(tune, train_baseline_setting, config["tuning_train_steps"])
            report["reference_measurements"] = {"evaluation": eval_reference, "training": train_reference,
                                                 "used_for_speed_selection": False}
            set_peft_model_state_dict(model, initial)

            def eval_search(settings, stage):
                settings = list(settings); rng.shuffle(settings)
                candidates = []
                for setting in settings:
                    set_peft_model_state_dict(model, initial)
                    result = evaluation(tune, setting)
                    quality = compare_predictions(eval_reference, result, config["logit_tolerance"])
                    entry = {"stage": stage, "settings": setting, "seconds": result["seconds"], "quality": quality,
                             "execution": result["execution"]}
                    candidates.append(entry); report["tuning"].append(entry)
                    write_report(args.output, report)
                    print(json.dumps({"stage": stage, "setting": setting, "seconds": entry["seconds"], "accepted": quality["accepted"]}), flush=True)
                allowed = [entry for entry in candidates if entry["quality"]["accepted"]]
                if not allowed:
                    raise RuntimeError("No evaluation setting met the numerical gate.")
                return min(allowed, key=lambda row: row["seconds"])["settings"]

            def train_search(settings, stage):
                settings = list(settings); rng.shuffle(settings)
                candidates = []
                for setting in settings:
                    warmup, _ = training(tune, setting, 1)
                    report["warmups"].append({"stage": stage, "settings": setting, **warmup})
                    result, state = training(tune, setting, config["tuning_train_steps"])
                    quality = train_agreement(train_reference_state, state)
                    entry = {"stage": stage, "settings": setting, **result, "quality": quality}
                    candidates.append(entry); report["tuning"].append(entry)
                    write_report(args.output, report)
                    print(json.dumps({"stage": stage, "setting": setting, "seconds": entry["seconds"], "accepted": quality["accepted"]}), flush=True)
                allowed = [entry for entry in candidates if entry["quality"]["accepted"]]
                if not allowed:
                    raise RuntimeError("No training setting met the numerical gate.")
                return min(allowed, key=lambda row: row["seconds"])["settings"]

            eval_thread = eval_search([{**eval_baseline_setting, "threads": t} for t in config["thread_candidates"]], "eval_threads")["threads"]
            selected_eval = eval_search([{**eval_baseline_setting, "threads": eval_thread}] + [{"threads": eval_thread, "batch_size": b, "sort_by_length": True} for b in config["eval_batch_candidates"]], "eval_batching")
            train_thread = train_search([{**train_baseline_setting, "threads": t} for t in config["thread_candidates"]], "train_threads")["threads"]
            selected_train = train_search([{"threads": train_thread, "microbatch_size": m} for m in config["microbatch_candidates"]], "train_microbatch")
            report["selected"] = {"evaluation": selected_eval, "training": selected_train, "confirmation_used_for_selection": False}
            for stage in ("evaluation", "training"):
                baseline_setting = eval_baseline_setting if stage == "evaluation" else train_baseline_setting
                candidate_setting = selected_eval if stage == "evaluation" else selected_train
                order_offset = rng.randrange(2)
                for repeat in range(1, config["confirmation_repeats"] + 1):
                    variants = ["baseline", "candidate"] if (repeat + order_offset) % 2 else ["candidate", "baseline"]
                    outcomes, states = {}, {}
                    for variant in variants:
                        setting = baseline_setting if variant == "baseline" else candidate_setting
                        set_peft_model_state_dict(model, initial)
                        if stage == "evaluation":
                            outcomes[variant] = evaluation(confirm, setting)
                        else:
                            outcomes[variant], states[variant] = training(confirm, setting, config["confirmation_train_steps"])
                    quality = compare_predictions(outcomes["baseline"], outcomes["candidate"], config["logit_tolerance"]) if stage == "evaluation" else train_agreement(states["baseline"], states["candidate"])
                    if stage == "training":
                        # Check updated model predictions on the remaining TRAIN-only confirmation records.
                        checks = {}
                        check_rows = confirm[config["confirmation_train_steps"] * 4:]
                        for variant in ("baseline", "candidate"):
                            set_peft_model_state_dict(model, states[variant])
                            checks[variant] = evaluation(check_rows, eval_baseline_setting)
                        output_check = compare_predictions(checks["baseline"], checks["candidate"], config["training_logit_tolerance"])
                        quality["updated_model_predictions"] = output_check
                        quality["accepted"] = quality["accepted"] and output_check["accepted"]
                    # Predictions are source IDs, scores and hashes only; no SMS text leaves the local source.
                    pair = {"repeat": repeat, "order": variants, "baseline_settings": baseline_setting,
                            "candidate_settings": candidate_setting, "baseline_seconds": outcomes["baseline"]["seconds"],
                            "candidate_seconds": outcomes["candidate"]["seconds"], "quality": quality,
                            "baseline": outcomes["baseline"], "candidate": outcomes["candidate"]}
                    report["confirmation"][stage].append(pair)
                    write_report(args.output, report)
                    print(json.dumps({"stage": "confirmation_" + stage, "repeat": repeat, "speedup": pair["baseline_seconds"] / pair["candidate_seconds"], "accepted": quality["accepted"]}), flush=True)
            report["summary"] = {stage: paired_statistics(values) for stage, values in report["confirmation"].items()}
            report["recommended"] = {}
            for stage in ("evaluation", "training"):
                summary = report["summary"][stage]
                baseline_setting = eval_baseline_setting if stage == "evaluation" else train_baseline_setting
                changed = report["selected"][stage] != baseline_setting
                accepted = changed and summary["all_quality_checks_pass"] and summary["faster_pairs"] == summary["pairs"] and summary["median_paired_speedup"] > 1.03
                report["recommended"][stage] = {"accepted": accepted,
                    "configuration_changed": changed, "settings": report["selected"][stage] if accepted else baseline_setting}
            report["cpu_evidence"] = {"torch_version": torch.__version__, "cuda_build": torch.version.cuda,
                                      "cuda_available": torch.cuda.is_available(), "parameter_devices": sorted({str(p.device) for p in model.parameters()}),
                                      "buffer_devices": sorted({str(b.device) for b in model.buffers()}), "dtype": "float32"}
            report["sampled_peak_process_rss_bytes"] = memory.peak_bytes
            report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        report["finished_utc"] = datetime.now(timezone.utc).isoformat()
        write_report(args.output, report)
    print(json.dumps({"status": report["status"], "recommended": report["recommended"], "summary": report["summary"]}), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/pipeline-speed.json"))
    parser.add_argument("--training-dir", type=Path, default=Path("artifacts/real-task-135m"))
    parser.add_argument("--output", type=Path, default=Path("results/pipeline-speed.json"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
