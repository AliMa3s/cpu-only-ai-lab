"""Validation-only threshold selection and immutable source-data checks."""
import copy
import json
from pathlib import Path
import tempfile

import pytest

from cpu_ai_lab.calibrate_classifier import (
    SCORE_KEY, THRESHOLDS, calibrate_report, run, select_threshold, validate_records,
)


def row(index, label, score):
    return {"id": index, "text_sha256": "hash-" + str(index), "expected": label,
            "predicted": int(score > 0.5), SCORE_KEY: score}


def source():
    validation = [row(1, 0, 0.6), row(2, 0, 0.7), row(3, 1, 0.9), row(4, 1, 0.95)]
    test = [row(5, 0, 0.55), row(6, 1, 0.85)]
    return {"model": {"id": "model", "revision": "revision"},
            "dataset": {"name": "sms", "archive_sha256": "archive", "source_file_sha256": "file", "split_seed": 4, "counts": {}, "split_manifest_sha256": "manifest"},
            "before": {"validation": {"records": copy.deepcopy(validation)}, "test": {"records": copy.deepcopy(test)}},
            "after": {"validation": {"records": copy.deepcopy(validation)}, "test": {"records": copy.deepcopy(test)}}}


def test_declared_grid_and_tie_break_choose_closest_half():
    assert THRESHOLDS == tuple(index / 20 for index in range(1, 20))
    selected = select_threshold([row(1, 0, 0.1), row(2, 1, 0.9)])
    assert selected["threshold"] == 0.5


def test_threshold_selection_uses_validation_and_test_changes_cannot_affect_it():
    evidence = source()
    unchanged = copy.deepcopy(evidence)
    first = calibrate_report(evidence, "source-a")
    assert first["after"]["selection"]["threshold"] == 0.7
    assert first["after"]["test"]["original"]["correct"] == 1
    assert first["after"]["test"]["threshold_selected_on_validation"]["correct"] == 2
    assert evidence == unchanged
    for phase in ("before", "after"):
        evidence[phase]["test"]["records"][0]["expected"] = 1
        evidence[phase]["test"]["records"][1]["expected"] = 0
    second = calibrate_report(evidence, "source-b")
    assert second["after"]["selection"] == first["after"]["selection"]
    assert second["after"]["test"]["threshold_selected_on_validation"]["correct"] == 0


def test_overlap_and_changed_before_after_labels_are_rejected():
    evidence = source()
    evidence["after"]["test"]["records"][0]["id"] = 1
    with pytest.raises(ValueError, match="overlap"):
        calibrate_report(evidence, "source")
    evidence = source()
    evidence["after"]["test"]["records"][0]["expected"] = 1
    with pytest.raises(ValueError, match="same held-out"):
        calibrate_report(evidence, "source")


def test_bad_scores_and_one_class_validation_rejected():
    for score in (float("nan"), float("inf"), -0.1, 1.1):
        with pytest.raises(ValueError, match="finite"):
            validate_records([row(1, 0, score)])
    with pytest.raises(ValueError, match="both labels"):
        select_threshold([row(1, 0, 0.1)])


def test_incomplete_run_refused_and_existing_report_preserved():
    with tempfile.TemporaryDirectory() as name:
        path = Path(name)
        (path / "run_status.json").write_text(json.dumps({"status": "running"}))
        with pytest.raises(ValueError, match="completed"):
            run(path, path / "report.json")
        (path / "report.json").write_text("preserve")
        with pytest.raises(FileExistsError):
            run(path, path / "report.json")
        assert (path / "report.json").read_text() == "preserve"
