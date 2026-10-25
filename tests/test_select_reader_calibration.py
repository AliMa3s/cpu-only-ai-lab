"""Selection integrity fixtures only; never load a model or run calibration."""
import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).parents[1] / "scripts" / "select_reader_calibration.py"
SPEC = importlib.util.spec_from_file_location("reader_calibration_selector", SCRIPT)
task = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(task)


def fixture(tmp_path, medians=(.8, .4, .6, .7)):
    root = Path(__file__).parents[1]
    original = task.read(root / "configs" / "reader.json")
    plan = task.read(root / "configs" / "reader-training-calibration.json")
    config_path, plan_path = tmp_path / "original.json", tmp_path / "plan.json"
    config_path.write_bytes(task.payload(original))
    plan_path.write_bytes(task.payload(plan))
    calibrations = tmp_path / "calibrations"
    calibrations.mkdir()
    planned = [[f"train-{step}-{index}" for index in range(8)] for step in range(8)]
    for (threads, micro), median in zip(task.CANDIDATES, medians, strict=True):
        config = {**original, "training_threads": threads, "evaluation_threads": threads}
        steps = [{"step": step, "warmup": step == 1, "seconds": 100 if step == 1 else median,
                  "learning_rate": original["learning_rate"] * step / original["warmup_steps"],
                  "global_batch_ids": planned[step - 1], "microbatch_order_ids": planned[step - 1],
                  "examples": 8, "microbatch_sizes": [micro] * (8 // micro), "microbatch_count": 8 // micro,
                  "global_padded_width": 32, "input_tokens": 192, "padded_tokens": 256,
                  "mean_loss": threads / 100} for step in range(1, 9)]
        result = {"format": "cpu-reader-training-calibration-v1", "config": config,
                  "threads": threads, "microbatch_size": micro, "declared_candidates": plan["candidates"],
                  "planned_global_batch_ids": planned, "steps": steps, "planned_updates": 8,
                  "completed_updates": 8, "warmup_updates": 1, "timed_updates": 7, "global_batch_size": 8,
                  "completed": True, "rss_below_budget": True, "eligible_for_selection": True,
                  "stop_reason": "eight_batches_complete", "median_timed_step_seconds": median,
                  "sampled_peak_process_rss_bytes": 2 * 1024**3, "sampled_peak_process_rss_gib": 2,
                  "input_tokens": 8 * 192, "padded_tokens": 8 * 256,
                  "reader_weights_sha256": original["reader_weights_sha256"],
                  "train_data_sha256": "a" * 64, "split_manifest_sha256": "b" * 64,
                  "ordered_calibration_ids_sha256": task.digest(json.dumps(planned, separators=(",", ":")).encode()),
                  "software_sha256": task.current_software(), "data_access": task.DATA_ACCESS,
                  "cpu": {"cuda_build": None, "cuda_available": False, "parameter_devices": ["cpu"], "optimizer_devices": ["cpu"]}}
        (calibrations / f"threads-{threads}-micro-{micro}.json").write_bytes(task.payload(result))
    return calibrations, plan_path, config_path, tmp_path / "selection.json", tmp_path / "optimized.json"


def test_selects_fastest_seven_step_median_and_changes_only_execution_settings(tmp_path):
    args = fixture(tmp_path)
    result = task.select(*args)
    assert result["selected"]["threads"] == 8 and result["selected"]["microbatch_size"] == 4
    assert result["selected"]["median_timed_step_seconds"] == .4
    original, optimized = task.read(args[2]), task.read(args[4])
    assert optimized["training_threads"] == 8 and optimized["training_microbatch_size"] == 4
    assert optimized["training_seconds_limit"] == 1800
    allowed = {"training_threads", "training_microbatch_size", "training_seconds_limit"}
    assert {key: value for key, value in optimized.items() if key not in allowed} == {
        key: value for key, value in original.items() if key not in allowed}
    assert optimized["evaluation_threads"] == original["evaluation_threads"]
    assert not result["validation_or_test_used_for_selection"] and not result["training_loss_used_for_selection"]
    assert task.select(*args) == result  # identical evidence is idempotent


def test_equal_medians_use_predeclared_candidate_order(tmp_path):
    result = task.select(*fixture(tmp_path, medians=(.4, .4, .5, .6)))
    assert result["selected"]["threads"] == 4 and result["selected"]["microbatch_size"] == 4


@pytest.mark.parametrize("change,match", [("median", "Median"), ("cpu", "CPU-only"),
                                         ("work", "global work"), ("code", "fingerprints"),
                                         ("eligibility", "eligibility")])
def test_rejects_inconsistent_or_unfair_candidate_evidence(tmp_path, change, match):
    args = fixture(tmp_path)
    path = args[0] / "threads-8-micro-4.json"
    value = task.read(path)
    if change == "median":
        value["median_timed_step_seconds"] = .001
    elif change == "cpu":
        value["cpu"]["cuda_available"] = True
    elif change == "work":
        value["steps"][0]["input_tokens"] += 1
        value["input_tokens"] += 1
    elif change == "code":
        value["software_sha256"]["reader_training.py"] = "c" * 64
    else:
        value["eligible_for_selection"] = False
    path.write_bytes(task.payload(value))
    with pytest.raises(ValueError, match=match):
        task.select(*args)
    assert not args[3].exists() and not args[4].exists()


def test_exact_rss_limit_is_ineligible_and_cannot_win(tmp_path):
    args = fixture(tmp_path)
    path = args[0] / "threads-8-micro-4.json"
    value = task.read(path)
    value.update(sampled_peak_process_rss_bytes=8 * 1024**3, sampled_peak_process_rss_gib=8,
                 rss_below_budget=False, eligible_for_selection=False)
    path.write_bytes(task.payload(value))
    result = task.select(*args)
    assert result["selected"]["threads"] == 12
    assert not result["candidates"][1]["eligible"]


def test_nonmatching_existing_output_is_preserved_before_any_other_write(tmp_path):
    args = fixture(tmp_path)
    args[4].write_bytes(b"existing configuration")
    with pytest.raises(FileExistsError, match="preserved"):
        task.select(*args)
    assert args[4].read_bytes() == b"existing configuration"
    assert not args[3].exists()
