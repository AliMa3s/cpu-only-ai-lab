"""Run and compare a complete CPU adaptation pipeline with tuned execution."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def execution_from_report(path: Path, config: dict) -> dict:
    report = read(path)
    if report.get("status") != "complete" or report.get("format") != "cpu-ai-lab-pipeline-speed-v1":
        raise ValueError("A completed pipeline-speed report is required.")
    source = report["source"]["configuration"]
    for key in ("model_id", "revision", "batch_size", "max_sequence_length", "learning_rate", "lora_rank", "lora_alpha", "lora_target_modules"):
        if source[key] != config[key]:
            raise ValueError(f"Tuning workload does not match this configuration: {key}")
    if config["batch_size"] != 4:
        raise ValueError("The measured training recipe requires global batch four.")
    train = report["recommended"]["training"]["settings"]
    evaluation = report["recommended"]["evaluation"]["settings"]
    for count in (train["threads"], evaluation["threads"], evaluation["batch_size"]):
        if type(count) is not int or not 1 <= count <= 256:
            raise ValueError("Invalid execution thread or batch count.")
    if type(train["microbatch_size"]) is not int or train["microbatch_size"] not in (1, 2, 4) or type(evaluation["sort_by_length"]) is not bool:
        raise ValueError("Invalid microbatch or sorting setting.")
    return {"report_sha256": digest(path), "training": train, "evaluation": evaluation,
            "changes": "Execution only: thread counts, evaluation batching/order, optional weighted microbatches. Same global updates and data."}


def schedule_hash(directory: Path, config: dict) -> str:
    """Reconstruct the baseline's recorded deterministic balanced sampler."""
    rows = [json.loads(line) for line in (directory / "data/train.jsonl").read_text(encoding="utf-8").splitlines()]
    rng = random.Random(config["seed"])
    indices = {label: [i for i, row in enumerate(rows) if row["label"] == label] for label in (0, 1)}
    for values in indices.values():
        rng.shuffle(values)
    positions, schedule = {0: 0, 1: 0}, []
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
        schedule.append([rows[index]["id"] for index in selected])
    return hashlib.sha256(json.dumps(schedule, separators=(",", ":")).encode()).hexdigest()


def validation_gate(baseline: dict, candidate: dict) -> dict:
    """Predeclared acceptance; this function receives validation metrics only."""
    if any(not math.isfinite(row[key]) for row in (baseline, candidate)
           for key in ("accuracy", "macro_f1", "binary_label_loss")):
        raise ValueError("Validation gate requires finite metrics.")
    checks = {"accuracy_no_lower": candidate["accuracy"] >= baseline["accuracy"],
              "macro_f1_no_lower": candidate["macro_f1"] >= baseline["macro_f1"],
              "binary_loss_within_absolute_0_01": candidate["binary_label_loss"] <= baseline["binary_label_loss"] + 0.01}
    return {"checks": checks, "accepted": all(checks.values()),
            "policy": "Validation accuracy and macro F1 must not decrease; binary-label loss may increase by at most 0.01 absolute. Test results never select settings."}


def compact(metric):
    return {key: metric[key] for key in ("examples", "correct", "accuracy", "macro_f1", "binary_label_loss", "seconds")}


def compare(baseline_dir: Path, candidate_dir: Path, output: Path):
    if output.exists():
        raise FileExistsError("Existing comparison reports are preserved.")
    runs, configs, curves = [], [], []
    for directory in (baseline_dir, candidate_dir):
        if read(directory / "run_status.json").get("status") != "complete":
            raise ValueError("Both runs must be complete.")
        runs.append(read(directory / "metrics.json"))
        configs.append(read(directory / "resolved_config.json"))
        curves.append(read(directory / "training_curve.json"))
    if {k: v for k, v in configs[0].items() if k != "execution"} != {k: v for k, v in configs[1].items() if k != "execution"}:
        raise ValueError("A semantic training configuration changed.")
    b, c = runs
    if b["dataset"]["split_manifest_sha256"] != c["dataset"]["split_manifest_sha256"]:
        raise ValueError("Data splits changed.")
    for name in ("optimizer_steps", "examples_seen_with_repetition", "unique_training_examples_seen", "input_tokens"):
        if b["training"][name] != c["training"][name]:
            raise ValueError(f"Training work changed: {name}")
    for cfg, curve in zip(configs, curves):
        expected_steps = list(range(1, cfg["optimizer_steps"] + 1))
        if [row["step"] for row in curve] != expected_steps:
            raise ValueError("Training curve does not contain every configured optimizer update.")
        expected_validation = [step for step in expected_steps if step % cfg["validation_every"] == 0 or step == cfg["optimizer_steps"]]
        if [row["step"] for row in curve if "validation" in row] != expected_validation:
            raise ValueError("Training curve does not contain every configured validation pass.")
    for a, d in zip(curves[0], curves[1], strict=True):
        if a["step"] != d["step"] or ("validation" in a) != ("validation" in d):
            raise ValueError("Optimizer/validation schedule changed.")
    schedules = [schedule_hash(path, cfg) for path, cfg in zip((baseline_dir, candidate_dir), configs)]
    if schedules[0] != schedules[1] or c["training"].get("sample_schedule_sha256") != schedules[1]:
        raise ValueError("Observed optimized sample schedule differs from the reconstructed baseline schedule.")
    for phase in ("before", "after"):
        for split in ("validation", "test"):
            identity = lambda rows: [(r["id"], r["text_sha256"], r["expected"], r["input_tokens"]) for r in rows]
            if identity(b[phase][split]["records"]) != identity(c[phase][split]["records"]):
                raise ValueError("Evaluation data, token counts, or record order changed.")
    gate = validation_gate(b["after"]["validation"], c["after"]["validation"])
    stages = []
    for metrics, curve in zip(runs, curves):
        evaluation = sum(row.get("validation", {}).get("seconds", 0) for row in curve)
        evaluation += sum(metrics[phase][split]["seconds"] for phase in ("before", "after") for split in ("validation", "test"))
        stages.append({"whole_run": metrics["elapsed_seconds"], "training_compute": metrics["training"]["compute_seconds_excluding_validation"],
                       "training_loop_including_validation": metrics["training"]["wall_seconds_including_validation"],
                       "all_evaluation": evaluation, "sampled_peak_process_rss_bytes": metrics["sampled_peak_process_rss_bytes"]})
    speedups = {key: stages[0][key] / stages[1][key] for key in ("whole_run", "training_compute", "training_loop_including_validation", "all_evaluation")}
    report = {"format": "cpu-ai-lab-whole-pipeline-comparison-v1", "status": "complete",
              "source_sha256": {"baseline_metrics": digest(baseline_dir / "metrics.json"), "optimized_metrics": digest(candidate_dir / "metrics.json")},
              "same_work": {"split_manifest_sha256": b["dataset"]["split_manifest_sha256"], "sample_schedule_sha256": schedules[0],
                            "baseline_schedule_evidence": "Reconstructed from original balanced-sampler algorithm, saved train records and seed; token/exposure totals checked.",
                            "candidate_schedule_evidence": "Hash recorded from actual sampled IDs in every update.",
                            "optimizer_steps": configs[0]["optimizer_steps"], "global_batch_size": configs[0]["batch_size"],
                            "input_tokens": b["training"]["input_tokens"], "periodic_validation_passes": sum("validation" in r for r in curves[0]),
                            "before_and_after_eval_examples": {split: b["after"][split]["examples"] for split in ("validation", "test")}},
              "baseline_seconds": stages[0], "optimized_seconds": stages[1], "speedups": speedups,
              "validation_acceptance": gate, "two_times_whole_run_target_met": gate["accepted"] and speedups["whole_run"] >= 2,
              "baseline": {"validation": compact(b["after"]["validation"]), "test": compact(b["after"]["test"]), "selected_step": b["selection"]["selected_step"]},
              "optimized": {"validation": compact(c["after"]["validation"]), "test": compact(c["after"]["test"]), "selected_step": c["selection"]["selected_step"], "execution": configs[1].get("execution")},
              "test_prediction_disagreements": sum(a["predicted"] != d["predicted"] for a, d in zip(b["after"]["test"]["records"], c["after"]["test"]["records"])),
              "cpu_evidence": {"baseline": b["cpu_evidence"], "optimized": c["cpu_evidence"]},
              "limitations": ["One fresh-process full run per variant, baseline first; not a repeated randomized whole-run benchmark.",
                              "elapsed_seconds starts inside _train after imports, initial configuration/status writes, and thread/seed setup; it excludes Python startup/imports and remaining report hashing/final metrics/status serialization, while including data processing, model loading, training, evaluation, and adapter/tokenizer saving.",
                              "Whole-run wall time is primary. The original baseline training-compute timer excludes collation; optimized training-compute includes collation and work counters, so that component ratio has different boundaries.",
                              "Filesystem/model caches were already populated. Other machine activity and thermal conditions were not controlled.",
                              "Repeated TRAIN-only microbenchmark confirmations are separate from this full-run measurement.",
                              "The historical SMS test set is reused for descriptive confirmation; it does not select optimizations.",
                              "Microbatching/thread changes may change floating-point trajectories; quality is measured rather than assumed."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    train = sub.add_parser("train")
    train.add_argument("--config", type=Path, default=Path("configs/real-task.json"))
    train.add_argument("--execution-report", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--archive", type=Path, default=Path(".cache/real-task/sms-spam-collection.zip"))
    comparison = sub.add_parser("compare")
    comparison.add_argument("--baseline", type=Path, required=True)
    comparison.add_argument("--candidate", type=Path, required=True)
    comparison.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.action == "train":
        from .real_task import train as train_model
        cfg = read(args.config)
        cfg["execution"] = execution_from_report(args.execution_report, cfg)
        train_model(cfg, args.output, args.archive)
    else:
        report = compare(args.baseline, args.candidate, args.output)
        print(json.dumps({"speedups": report["speedups"], "validation": report["validation_acceptance"], "test_disagreements": report["test_prediction_disagreements"], "target_met": report["two_times_whole_run_target_met"]}, indent=2))


if __name__ == "__main__":
    main()
