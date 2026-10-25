import copy
import json

import pytest

from cpu_ai_lab.pipeline_run import compare, execution_from_report, schedule_hash, validation_gate


def test_validation_gate_requires_all_three_measures():
    baseline = {"accuracy": 0.95, "macro_f1": 0.9, "binary_label_loss": 0.1}
    assert validation_gate(baseline, dict(baseline))["accepted"]
    for field, value in (("accuracy", 0.94), ("macro_f1", 0.89), ("binary_label_loss", 0.111)):
        candidate = {**baseline, field: value}
        assert not validation_gate(baseline, candidate)["accepted"]


@pytest.mark.parametrize("nonfinite", [float("nan"), float("inf"), -float("inf")])
def test_validation_gate_rejects_nonfinite_metrics(nonfinite):
    baseline = {"accuracy": 0.95, "macro_f1": 0.9, "binary_label_loss": 0.1}
    with pytest.raises(ValueError, match="finite"):
        validation_gate(baseline, {**baseline, "macro_f1": nonfinite})


def test_execution_recipe_rejects_changed_workload(tmp_path):
    config = {"model_id": "m", "revision": "r", "batch_size": 4, "max_sequence_length": 192,
              "learning_rate": 0.0005, "lora_rank": 8, "lora_alpha": 16, "lora_target_modules": ["q_proj", "v_proj"]}
    report = {"status": "complete", "format": "cpu-ai-lab-pipeline-speed-v1", "source": {"configuration": config},
              "recommended": {"training": {"settings": {"threads": 6, "microbatch_size": 2}},
                              "evaluation": {"settings": {"threads": 12, "batch_size": 16, "sort_by_length": True}}}}
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(report))
    assert execution_from_report(path, config)["training"]["microbatch_size"] == 2
    wrong = copy.deepcopy(config)
    wrong["learning_rate"] *= 2
    with pytest.raises(ValueError, match="learning_rate"):
        execution_from_report(path, wrong)
    report["status"] = "running"
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="completed"):
        execution_from_report(path, config)
    report["status"] = "complete"
    report["recommended"]["training"]["settings"]["microbatch_size"] = True
    path.write_text(json.dumps(report))
    with pytest.raises(ValueError, match="microbatch"):
        execution_from_report(path, config)


def write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


@pytest.fixture
def complete_run_pair(tmp_path):
    """Two tiny on-disk reports; no model, tensor or dataset downloads."""
    baseline, candidate = tmp_path / "baseline", tmp_path / "candidate"
    config = {"model_id": "fixture", "revision": "fixed", "seed": 23,
              "optimizer_steps": 2, "validation_every": 2, "batch_size": 4,
              "threads": 4, "max_sequence_length": 16}
    train_rows = [{"id": f"train-{i}", "label": i % 2} for i in range(4)]

    def evaluation(split):
        return {"examples": 2, "correct": 2, "accuracy": 1.0, "macro_f1": 1.0,
                "binary_label_loss": 0.2, "seconds": 1.0,
                "records": [{"id": f"{split}-{i}", "text_sha256": f"text-{split}-{i}",
                             "expected": i, "predicted": i, "input_tokens": 4} for i in range(2)]}

    for directory, optimized in ((baseline, False), (candidate, True)):
        (directory / "data").mkdir(parents=True)
        (directory / "data/train.jsonl").write_text("".join(json.dumps(row) + "\n" for row in train_rows), encoding="utf-8")
        cfg = {**config, **({"execution": {"training": {"threads": 2, "microbatch_size": 4}}} if optimized else {})}
        metrics = {"dataset": {"split_manifest_sha256": "same-split"},
                   "training": {"optimizer_steps": 2, "examples_seen_with_repetition": 8,
                                "unique_training_examples_seen": 4, "input_tokens": 32,
                                "compute_seconds_excluding_validation": 2.0 if optimized else 4.0,
                                "wall_seconds_including_validation": 3.0 if optimized else 5.0},
                   "before": {split: evaluation(split) for split in ("validation", "test")},
                   "after": {split: evaluation(split) for split in ("validation", "test")},
                   "elapsed_seconds": 8.0 if optimized else 16.0,
                   "sampled_peak_process_rss_bytes": 1024,
                   "selection": {"selected_step": 2},
                   "cpu_evidence": {"torch_cuda_build": None, "cuda_available": False, "parameter_devices": ["cpu"]}}
        if optimized:
            metrics["training"]["sample_schedule_sha256"] = schedule_hash(directory, cfg)
        write_json(directory / "resolved_config.json", cfg)
        write_json(directory / "run_status.json", {"status": "complete"})
        write_json(directory / "training_curve.json", [{"step": 1}, {"step": 2, "validation": {"seconds": 1.0}}])
        write_json(directory / "metrics.json", metrics)
    return baseline, candidate, tmp_path / "comparison.json"


def edit_json(path, edit):
    value = json.loads(path.read_text())
    edit(value)
    write_json(path, value)


def test_compare_accepts_same_work_and_reports_whole_run_gate(complete_run_pair):
    baseline, candidate, output = complete_run_pair
    report = compare(baseline, candidate, output)
    assert report["validation_acceptance"]["accepted"]
    assert report["speedups"]["whole_run"] == 2.0
    assert report["two_times_whole_run_target_met"]
    assert report["same_work"]["periodic_validation_passes"] == 1
    assert report["same_work"]["sample_schedule_sha256"] == schedule_hash(baseline, json.loads((baseline / "resolved_config.json").read_text()))
    assert report["test_prediction_disagreements"] == 0
    assert json.loads(output.read_text()) == report
    with pytest.raises(FileExistsError, match="preserved"):
        compare(baseline, candidate, output)


@pytest.mark.parametrize("change", ["config", "split", "input_tokens", "schedule", "record_identity"])
def test_compare_rejects_changed_work(complete_run_pair, change):
    baseline, candidate, output = complete_run_pair
    if change == "config":
        edit_json(candidate / "resolved_config.json", lambda value: value.update(seed=24))
    elif change == "split":
        edit_json(candidate / "metrics.json", lambda value: value["dataset"].update(split_manifest_sha256="other-split"))
    elif change == "input_tokens":
        edit_json(candidate / "metrics.json", lambda value: value["training"].update(input_tokens=31))
    elif change == "schedule":
        edit_json(candidate / "metrics.json", lambda value: value["training"].update(sample_schedule_sha256="wrong-order"))
    else:
        edit_json(candidate / "metrics.json", lambda value: value["after"]["test"]["records"][0].update(text_sha256="changed-message"))
    with pytest.raises(ValueError):
        compare(baseline, candidate, output)
    assert not output.exists()


@pytest.mark.parametrize("change", ["missing_update", "missing_validation", "extra_validation"])
def test_compare_rejects_incomplete_or_changed_validation_schedule(complete_run_pair, change):
    baseline, candidate, output = complete_run_pair
    curve = [{"step": 1}, {"step": 2, "validation": {"seconds": 1.0}}]
    if change == "missing_update":
        curve.pop(0)
    elif change == "missing_validation":
        del curve[1]["validation"]
    else:
        curve[0]["validation"] = {"seconds": 1.0}
    write_json(candidate / "training_curve.json", curve)
    with pytest.raises(ValueError, match="configured"):
        compare(baseline, candidate, output)
    assert not output.exists()


def test_compare_validation_failure_blocks_target_even_with_fast_run(complete_run_pair):
    baseline, candidate, output = complete_run_pair
    edit_json(candidate / "metrics.json", lambda value: value["after"]["validation"].update(binary_label_loss=0.22))
    report = compare(baseline, candidate, output)
    assert report["speedups"]["whole_run"] == 2.0
    assert not report["validation_acceptance"]["accepted"]
    assert not report["two_times_whole_run_target_met"]


def test_compare_test_predictions_do_not_select_acceptance(complete_run_pair):
    baseline, candidate, output = complete_run_pair

    def change_test(value):
        test = value["after"]["test"]
        test.update(correct=1, accuracy=0.5, macro_f1=1 / 3, binary_label_loss=2.0)
        test["records"][0]["predicted"] = 1

    edit_json(candidate / "metrics.json", change_test)
    report = compare(baseline, candidate, output)
    assert report["validation_acceptance"]["accepted"]
    assert report["two_times_whole_run_target_met"]
    assert report["test_prediction_disagreements"] == 1
