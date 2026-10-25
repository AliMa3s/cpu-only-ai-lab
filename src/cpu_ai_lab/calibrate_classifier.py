"""Select an SMS classification threshold using saved validation scores only.

No model execution or further training is performed. Original training metrics
are preserved. Threshold selection changes the decision rule; it does not make
the model's two-label softmax scores calibrated probabilities.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from .real_task import classification_metrics, predict_local

THRESHOLDS = tuple(round(index / 20, 2) for index in range(1, 20))
SCORE_KEY = "spam_probability_conditioned_on_two_labels"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def records_hash(records: list[dict]) -> str:
    """Hash only the evidence used in threshold decisions/scoring, canonically."""
    selected = [{key: record[key] for key in ("id", "text_sha256", "expected", "predicted", SCORE_KEY)} for record in records]
    return sha256(json.dumps(selected, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8"))


def validate_records(records: list[dict]) -> None:
    if not records:
        raise ValueError("Classification records must be nonempty.")
    identifiers = []
    for record in records:
        if record["expected"] not in (0, 1) or record["predicted"] not in (0, 1):
            raise ValueError("Expected binary classification labels.")
        value = record[SCORE_KEY]
        if not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("Scores must be finite values in [0,1].")
        identifiers.append(record["id"])
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("Repeated example identifiers in classification records.")


def metrics_at_threshold(records: list[dict], threshold: float) -> dict:
    return classification_metrics([record["expected"] for record in records], [int(record[SCORE_KEY] > threshold) for record in records])


def select_threshold(validation_records: list[dict]) -> dict:
    """This function cannot read test data; it accepts validation evidence only."""
    validate_records(validation_records)
    if {record["expected"] for record in validation_records} != {0, 1}:
        raise ValueError("Threshold selection requires both labels in validation.")
    candidates = [{"threshold": threshold, **metrics_at_threshold(validation_records, threshold)} for threshold in THRESHOLDS]
    # Rounding makes symmetric distances (e.g. .45/.55) exact for tie-breaking.
    selected = max(candidates, key=lambda row: (row["macro_f1"], -round(abs(row["threshold"] - 0.5), 8), -row["threshold"]))
    return {"threshold": selected["threshold"], "selected_validation_metrics": selected,
            "candidates": candidates, "validation_records_sha256": records_hash(validation_records),
            "rule": "maximize validation macro F1; ties closest to 0.5, then lower threshold"}


def score_locked_threshold(records: list[dict], threshold: float) -> dict:
    validate_records(records)
    original = classification_metrics([record["expected"] for record in records], [record["predicted"] for record in records])
    predictions = [{"id": record["id"], "text_sha256": record["text_sha256"], "expected": record["expected"],
                    "original_predicted": record["predicted"], "predicted": int(record[SCORE_KEY] > threshold),
                    SCORE_KEY: record[SCORE_KEY]} for record in records]
    changed = sum(row["predicted"] != row["original_predicted"] for row in predictions)
    return {"threshold": threshold, "original": original, "threshold_selected_on_validation": metrics_at_threshold(records, threshold),
            "changed_decisions": changed, "records_sha256": records_hash(records), "records": predictions}


def calibrate_report(metrics: dict, source_hash: str) -> dict:
    # Lock both thresholds before reading the test records, including the
    # pretrained baseline's threshold so comparisons receive the same treatment.
    selections = {phase: select_threshold(metrics[phase]["validation"]["records"]) for phase in ("before", "after")}
    results = {}
    for phase in ("before", "after"):
        validation = metrics[phase]["validation"]["records"]
        test = metrics[phase]["test"]["records"]
        validate_records(test)
        if {row["id"] for row in validation} & {row["id"] for row in test}:
            raise ValueError("Validation and test identifiers overlap.")
        if {row["text_sha256"] for row in validation} & {row["text_sha256"] for row in test}:
            raise ValueError("Validation and test text hashes overlap.")
        results[phase] = {"selection": selections[phase],
                          "validation": score_locked_threshold(validation, selections[phase]["threshold"]),
                          "test": score_locked_threshold(test, selections[phase]["threshold"])}
    for split in ("validation", "test"):
        identities = [{(record["id"], record["text_sha256"], record["expected"]) for record in metrics[phase][split]["records"]} for phase in ("before", "after")]
        if identities[0] != identities[1]:
            raise ValueError("Before/after records do not describe the same held-out examples.")
    return {
        "experiment": "Validation-selected decision thresholds for CPU SMS classifiers",
        "completed_utc": datetime.now(timezone.utc).isoformat(), "source_metrics_sha256": source_hash,
        "model": metrics["model"], "dataset": {key: metrics["dataset"][key] for key in ("name", "archive_sha256", "source_file_sha256", "split_seed", "counts", "split_manifest_sha256")},
        "protocol": {"threshold_grid": list(THRESHOLDS), "positive_label": 1, "decision": "spam iff two-label softmax score is strictly greater than the locked threshold",
                     "selection": "Independent BEFORE and AFTER thresholds chosen by validation macro F1, ties closest to 0.5 then lower threshold",
                     "test_used_for_threshold_selection": False, "original_metrics_modified": False,
                     "thresholds_locked_before_test_scoring": True},
        **results,
        "limitations": ["Threshold selection improves a decision rule; the scores are still not calibrated deployment probabilities.",
                        "This is a follow-up analysis, not a preregistered blind study. The grid and tie rule were declared after observing early validation behavior and before inspecting the final trained-model test results; the initial pretrained-model test summary had already been seen. The threshold-selection algorithm uses validation records only.",
                        "Thresholds may not transfer to different class prevalence, current SMS traffic, or another dataset.",
                        "Validation threshold tuning adds model-selection degrees of freedom; the original fixed-threshold results remain primary and preserved.",
                        "Historical dataset contamination and template grouping limitations from the source experiment still apply."],
    }


def run(training_dir: Path, output: Path) -> dict:
    if output.exists():
        raise FileExistsError("Refusing to replace an existing calibration report.")
    status = json.loads((training_dir / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError("Only completed training runs can be calibrated.")
    source = (training_dir / "metrics.json").read_bytes()
    metrics = json.loads(source)
    report = calibrate_report(metrics, sha256(source))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return report


def predict_with_threshold(training_dir: Path, report_path: Path, text: str, threads: int = 4) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if sha256((training_dir / "metrics.json").read_bytes()) != report["source_metrics_sha256"]:
        raise ValueError("Calibration report does not match this training run's metrics hash.")
    threshold = report["after"]["selection"]["threshold"]
    if threshold not in THRESHOLDS:
        raise ValueError("Saved threshold is not in the declared selection grid.")
    prediction = predict_local(training_dir, text, threads)
    return {**prediction, "original_label": prediction["label"],
            "label": "spam" if prediction["spam_score"] > threshold else "ham",
            "threshold_selected_on_validation": threshold}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--predict")
    parser.add_argument("--threads", type=int, default=4)
    args = parser.parse_args()
    if args.predict is not None:
        if args.report is None:
            parser.error("--predict requires --report")
        print(json.dumps(predict_with_threshold(args.training_dir, args.report, args.predict, args.threads), indent=2))
        return
    if args.output is None:
        parser.error("--output is required to create a calibration report")
    report = run(args.training_dir, args.output)
    print(json.dumps({phase: {"threshold": report[phase]["selection"]["threshold"],
                              "original_test": report[phase]["test"]["original"],
                              "calibrated_test": report[phase]["test"]["threshold_selected_on_validation"]} for phase in ("before", "after")}, indent=2))


if __name__ == "__main__":
    main()
