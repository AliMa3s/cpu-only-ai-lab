"""Derive historical stage times and padding work without running a model.

The optional training-padding pass loads only the cached tokenizer and local
training records. No model weights are loaded and no model forward pass runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def padding_work(lengths: list[int], batch_size: int, sort_lengths: bool) -> dict:
    if not lengths or batch_size < 1 or any(length < 1 for length in lengths):
        raise ValueError("Positive sequence lengths and batch size are required")
    ordered = sorted(lengths) if sort_lengths else list(lengths)
    batches = [ordered[i:i + batch_size] for i in range(0, len(ordered), batch_size)]
    tokens = sum(lengths)
    padded = sum(len(batch) * max(batch) for batch in batches)
    attention_proxy = sum(len(batch) * max(batch) ** 2 for batch in batches)
    return {
        "batch_size": batch_size, "ordering": "length_sorted" if sort_lengths else "original",
        "examples": len(lengths), "batches": len(batches), "unpadded_tokens": tokens,
        "padded_token_slots": padded, "padding_token_slots": padded - tokens,
        "padding_fraction": (padded - tokens) / padded,
        "sum_batch_sequence_squares": attention_proxy,
        "interpretation": "Token slots approximate linear-layer work; sequence squares are only an attention-shape proxy, not measured FLOPs or runtime.",
    }


def training_schedule(labels: list[int], config: dict) -> list[list[int]]:
    """Reproduce real_task's balanced sampler, including its shuffle order."""
    if config["batch_size"] % 2:
        raise ValueError("The historical sampler requires an even batch size")
    rng = random.Random(config["seed"])
    indices = {label: [i for i, value in enumerate(labels) if value == label] for label in (0, 1)}
    if not all(indices.values()):
        raise ValueError("Both labels are required")
    for values in indices.values():
        rng.shuffle(values)
    positions = {0: 0, 1: 0}
    schedule = []
    for _ in range(config["optimizer_steps"]):
        selected = []
        for label in (0, 1):
            for _ in range(config["batch_size"] // 2):
                if positions[label] == len(indices[label]):
                    rng.shuffle(indices[label])
                    positions[label] = 0
                selected.append(indices[label][positions[label]])
                positions[label] += 1
        rng.shuffle(selected)
        schedule.append(selected)
    return schedule


def profile_one(results: Path, size: str, training_dir: Path | None = None) -> dict:
    metrics_path = results / f"real-task-{size}.json"
    curve_path = results / f"real-task-{size}-curve.json"
    config_path = results / f"real-task-{size}-config.json"
    metrics, curve, config = (read_json(path) for path in (metrics_path, curve_path, config_path))
    compute = metrics["training"]["compute_seconds_excluding_validation"]
    periodic = sum(row.get("validation", {}).get("seconds", 0) for row in curve)
    before = sum(stage["seconds"] for stage in metrics["before"].values())
    after = sum(stage["seconds"] for stage in metrics["after"].values())
    total = metrics["elapsed_seconds"]
    evaluation = periodic + before + after
    residual = total - compute - evaluation
    if residual < -0.1:
        raise ValueError("Historical component times exceed total runtime")
    fractions = {"training_compute": compute / total, "evaluation": evaluation / total,
                 "other": max(residual, 0) / total}
    remaining_for_eval = total / 2 - compute - residual
    profile = {
        "model": metrics["model"]["id"],
        "source_hashes": {path.name: file_hash(path) for path in (metrics_path, curve_path, config_path)},
        "stage_seconds": {"training_compute": compute, "periodic_validation": periodic,
                          "before_evaluation": before, "after_evaluation": after,
                          "all_evaluation": evaluation, "other_by_subtraction": residual, "whole_run": total},
        "fractions_of_whole_run": fractions,
        "amdahl_examples": {
            "evaluation_2x_training_unchanged_whole_run_speedup": total / (compute + evaluation / 2 + residual),
            "training_2x_evaluation_unchanged_whole_run_speedup": total / (compute / 2 + evaluation + residual),
            "both_2x_whole_run_speedup": total / (compute / 2 + evaluation / 2 + residual),
            "required_eval_speedup_for_2x_whole_run_if_training_unchanged": evaluation / remaining_for_eval if remaining_for_eval > 0 else None,
            "note": "Arithmetic scenarios, not measured achievable speedups. They assume fixed work and unchanged other overhead.",
        },
        "evaluation_padding": {},
    }
    for split in ("validation", "test"):
        before_rows, after_rows = metrics["before"][split]["records"], metrics["after"][split]["records"]
        identities = lambda rows: [(r["id"], r["text_sha256"], r["input_tokens"]) for r in rows]
        if identities(before_rows) != identities(after_rows):
            raise ValueError("Before/after evaluation rows do not describe identical input work")
        lengths = [row["input_tokens"] for row in before_rows]
        baseline = padding_work(lengths, config["evaluation_batch_size"], False)
        candidates = []
        for batch_size in (4, 8, 16, 32, 64):
            for sorted_order in (False, True):
                candidate = padding_work(lengths, batch_size, sorted_order)
                candidate["padded_slot_ratio_vs_historical"] = candidate["padded_token_slots"] / baseline["padded_token_slots"]
                candidates.append(candidate)
        profile["evaluation_padding"][split] = {"historical": baseline, "candidate_shape_work": candidates,
                                                 "number_of_passes": 2 + (len([r for r in curve if "validation" in r]) if split == "validation" else 0)}
    if training_dir is not None:
        # Import only for this optional tokenizer-only calculation. No weights.
        from transformers import AutoTokenizer
        from cpu_ai_lab.real_task import encode_split
        rows = [json.loads(line) for line in (training_dir / "data/train.jsonl").read_text(encoding="utf-8").splitlines()]
        tokenizer = AutoTokenizer.from_pretrained(training_dir / "tokenizer", local_files_only=True, trust_remote_code=False)
        encoded = encode_split(tokenizer, rows, config["max_sequence_length"])
        lengths = [len(row["input_ids"]) for row in encoded]
        schedule = training_schedule([row["label"] for row in rows], config)
        seen_tokens = sum(lengths[i] for batch in schedule for i in batch)
        if seen_tokens != metrics["training"]["input_tokens"]:
            raise ValueError("Reconstructed training work differs from recorded token count")
        batches = [[lengths[i] for i in batch] for batch in schedule]
        padded = sum(len(batch) * max(batch) for batch in batches)
        schedule_ids = [[rows[index]["id"] for index in batch] for batch in schedule]
        profile["training_padding"] = {
            "optimizer_steps": len(batches), "global_batch_size": config["batch_size"],
            "examples_seen": sum(map(len, batches)), "unique_examples_seen": len({i for batch in schedule for i in batch}),
            "input_tokens": seen_tokens, "padded_token_slots": padded,
            "padding_fraction": (padded - seen_tokens) / padded,
            "sample_id_schedule_sha256": hashlib.sha256(json.dumps(schedule_ids, separators=(",", ":")).encode()).hexdigest(),
            "microbatch_shape_candidates": [],
        }
        for microbatch in (1, 2, 4):
            slots = 0
            for batch in batches:
                for offset in range(0, len(batch), microbatch):
                    sub = batch[offset:offset + microbatch]
                    slots += len(sub) * max(sub)
            profile["training_padding"]["microbatch_shape_candidates"].append({
                "microbatch_size": microbatch, "global_batch_size_unchanged": config["batch_size"],
                "forward_backward_calls": sum(math.ceil(len(batch) / microbatch) for batch in batches),
                "padded_token_slots": slots, "padded_slot_ratio_vs_historical": slots / padded,
                "note": "Microbatch gradients must be weighted by microbatch examples/global-batch examples before a single clipping and optimizer step. Extra calls can erase padding savings.",
            })
    return profile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("results/pipeline-profile.json"))
    parser.add_argument("--include-training-padding", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Choose a fresh profile path; existing evidence is preserved")
    report = {
        "format": "cpu-ai-lab-historical-pipeline-profile-v1",
        "method": "Derived from recorded timers and input lengths; no model weights loaded or forward passes executed. Optional training pass loads only the cached tokenizer and recreates the recorded sampler.",
        "limitations": ["Historical wall times are not a fresh paired benchmark.",
                        "Padding/token-square counts are workload-shape estimates, not actual hardware FLOPs or expected speedups.",
                        "Evaluation must retain all examples and original scoring rules when changing batch shape.",
                        "Batching, microbatching and CPU thread changes can alter floating-point results; confirm validation quality before reporting held-out test quality.",
                        "The derived residual includes loading, preprocessing, artifact writes, sampler/Python overhead, and timer boundaries; it is not a separately profiled function."],
        "profiles": {size: profile_one(args.results, size, Path(f"artifacts/real-task-{size}") if args.include_training_padding else None) for size in ("135m", "360m")},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({size: {"stage_seconds": row["stage_seconds"], "fractions": row["fractions_of_whole_run"],
                            "amdahl": row["amdahl_examples"], "training_padding": row.get("training_padding")}
                      for size, row in report["profiles"].items()}, indent=2))


if __name__ == "__main__":
    main()
