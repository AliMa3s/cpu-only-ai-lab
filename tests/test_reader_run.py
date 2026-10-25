"""Mocked orchestration with a one-parameter CPU model and no pretrained loads."""
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cpu_ai_lab import reader_run as task


def configuration():
    config = task.read(Path(__file__).parents[1] / "configs/reader.json")
    config.update(train_size=8, validation_size=4, test_size=4, training_batch_size=4,
                  optimizer_steps=2, warmup_steps=1, training_threads=1, evaluation_threads=1)
    return config


def fixture_rows(split, count=4):
    return [{"id": f"{split}-{i}", "context": "Paris is a city.", "question": "Which city?",
             "is_impossible": i >= count // 2, "references": [] if i >= count // 2 else ["Paris"],
             "length_group": "short" if i % 2 == 0 else "long", "context_tokens": 4 if i % 2 == 0 else 200,
             "reader_tokens": 8 + i, "prompt_tokens": 10 + i,
             "literal_answer_spans": [] if i >= count // 2 else [{"start": 0, "end": 5, "text": "Paris"}]}
            for i in range(count)]


def test_one_pass_schedule_is_balanced_complete_deterministic():
    rows, config = fixture_rows("train", 8), configuration()
    result = task.schedule(rows, config)
    assert result == task.schedule(rows, config)
    assert sorted(i for batch in result for i in batch) == list(range(8))
    assert len(result) == 2 and all(len(batch) == 4 for batch in result)
    assert all(sum(rows[i]["is_impossible"] for i in batch) == 2 for batch in result)
    rows[0]["is_impossible"] = True
    with pytest.raises(ValueError, match="balanced"):
        task.schedule(rows, config)
    with pytest.raises(ValueError, match="exactly one"):
        task.validate_config({**config, "optimizer_steps": 3})


def selection(*, accepted=True, f1=.7, recall=.4, retention=1.0, constrained=True, threshold=0):
    return {"accepted": accepted, "selected": {"answerable_token_f1": f1, "correct_abstention_recall": recall},
            "selected_threshold": threshold if accepted else None, "diagnostic_threshold": threshold,
            "diagnostic_candidate": {"gate": {"retention_and_f1_pass": constrained, "exact_correct_retention": retention},
                                     "metrics": {"answerable_token_f1": f1, "correct_abstention_recall": recall}}}


def test_two_checkpoint_choice_uses_quality_then_recall_and_initial_on_tie():
    assert task.choose_checkpoint({"initial": selection(), "final": selection()})["checkpoint"] == "initial"
    assert task.choose_checkpoint({"initial": selection(f1=.8), "final": selection(f1=.7, recall=1)})["checkpoint"] == "initial"
    assert task.choose_checkpoint({"initial": selection(), "final": selection(recall=.6)})["checkpoint"] == "final"
    assert task.choose_checkpoint({"initial": selection(accepted=False, f1=1), "final": selection()})["checkpoint"] == "final"
    failed = task.choose_checkpoint({"initial": selection(accepted=False, constrained=True, recall=.2),
                                    "final": selection(accepted=False, constrained=False, recall=1)})
    assert failed["kind"] == "generator" and failed["checkpoint"] is None and failed["threshold"] is None
    assert failed["diagnostic_checkpoint"] == "initial"


class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(.25))

    def forward(self, **inputs):
        return SimpleNamespace(loss=(self.weight - 1).square())


def mocked_run(tmp_path, monkeypatch, *, guard="none", microbatch=None):
    config = configuration()
    if microbatch is not None:
        config["training_microbatch_size"] = microbatch
    if guard == "time":
        config["training_seconds_limit"] = 1
    splits = {name: fixture_rows(name, 8 if name == "train" else 4) for name in ("train", "validation", "test")}
    prepared, source, output = tmp_path / "data", tmp_path / "source", tmp_path / "run"
    prepared.mkdir(); source.mkdir()
    task.write_json(prepared / "split_manifest.json", {"fixture": True})
    task.write_json(source / "metrics.json", {"dataset": {"split_manifest_sha256": "original-manifest"}})
    provenance = {"prior_manifest_sha256": ["original-manifest"]}
    source_evidence = {"metrics_sha256": task.digest(source / "metrics.json"), "fixture": True}
    saved, events = {}, []

    class Memory:
        peak_bytes = 9 * 1024**3 if guard == "rss" else 1024
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass

    monkeypatch.setattr(task, "PeakRSS", Memory)
    monkeypatch.setattr(task, "setup", lambda config: None)
    monkeypatch.setattr(task.torch, "set_num_threads", lambda count: None)
    monkeypatch.setattr(task.data, "load_prepared", lambda *args: (splits, provenance))
    monkeypatch.setattr(task, "initializer_path", lambda *args: tmp_path / "unused-initializer")
    monkeypatch.setattr(task, "load_generator", lambda *args: (TinyModel().requires_grad_(False), None, 1, source_evidence))
    monkeypatch.setattr(task, "adapter_snapshot", lambda model: {})
    monkeypatch.setattr(task.reader, "load_initial", lambda *args, **kwargs: (TinyModel(), SimpleNamespace(pad_token_id=0)))

    def observe(label):
        if "test" in label:
            assert (output / "policy.json").exists()
            policy = task.read(output / "policy.json")
            selected = task.read(output / "selection.json")
            assert task.choose_checkpoint(selected["checkpoints"]) == selected["choice"]
            assert policy["selection_sha256"] == task.digest(output / "selection.json")
            assert policy["selection_uses_test"] is False
        else:
            assert not (output / "policy.json").exists()
        events.append(label)

    def generation(model, tokenizer, rows, *args):
        observe("generator_" + rows[0]["id"].split("-")[0])
        records = [task.scoring.score_prediction(row, "Paris") for row in rows]
        return {**task.scoring.summarize(records), "records": records}, [{"id": row["id"], "prediction": "Paris"} for row in rows]

    monkeypatch.setattr(task.generator, "generation_evaluation", generation)

    def evaluate(model, tokenizer, rows, config, label):
        observe(label)
        # A useful initializer followed by worse adaptation must retain initial.
        bad = label.startswith("final")
        return [{"id": row["id"], "span_prediction": "Rome" if bad else "Paris",
                 "null_minus_span_score": 1 if row["is_impossible"] else -1,
                 "input_tokens": row["reader_tokens"]} for row in rows], .01

    monkeypatch.setattr(task, "reader_evaluation", evaluate)
    monkeypatch.setattr(task, "loss_evaluation", lambda model, tokenizer, rows, config:
                        {"mean_span_loss": float((model.weight.detach() - 1).square()), "examples": len(rows)})
    monkeypatch.setattr(task, "encoded_rows", lambda rows, *args:
                        ([{"input_ids": [1, 2], "attention_mask": [1, 1]} for row in rows], [(0, 0)] * len(rows)))

    def save(path, model, tokenizer, metadata):
        path.mkdir()
        saved[path.name] = copy.deepcopy(model.state_dict())
        manifest = {"checkpoint": metadata["checkpoint"]}
        task.write_json(path / "reader_manifest.json", manifest)
        return manifest

    def load(path):
        model = TinyModel()
        model.load_state_dict(saved[path.name])
        return model, SimpleNamespace(pad_token_id=0)

    monkeypatch.setattr(task.reader, "save_reader", save)
    monkeypatch.setattr(task.reader, "load_reader", load)
    if guard == "time":
        clock = iter(range(0, 10000, 2))
        monkeypatch.setattr(task.time, "perf_counter", lambda: next(clock))
    task.run(config, output, prepared, source)
    assert events == ["generator_validation", "initial_reader_validation", "final_reader_validation",
                      "generator_test", "initial_reader_test", "final_reader_test"]
    assert task.read(output / "run_status.json")["status"] == "complete"
    return output


@pytest.fixture
def completed(tmp_path, monkeypatch):
    return mocked_run(tmp_path, monkeypatch)


def test_full_mock_run_trains_once_and_freezes_choice_before_all_test_predictions(completed):
    policy, report = task.load_policy(completed)
    assert policy["kind"] == "reader" and policy["checkpoint"] == "initial"
    assert report["training"]["completed_updates"] == 2
    assert report["training"]["unique_examples_seen"] == report["training"]["presentations"] == 8
    assert report["training"]["weights_changed"]
    assert report["training"]["trainable_parameters"] == 1
    assert report["cpu"]["reader_parameter_devices"] == report["cpu"]["optimizer_devices"] == ["cpu"]
    assert report["test"]["selected"]["paired_retention"]["exact_correct_retention"] == 1
    assert report["test"]["final"]["threshold_selected"] is None
    assert report["selection"]["test_used_for_selection"] is False


def test_microbatch_runner_keeps_one_update_per_complete_global_batch(tmp_path, monkeypatch):
    output = mocked_run(tmp_path, monkeypatch, microbatch=2)
    report = task.read(output / "report.json")
    curve = task.read(output / "training_curve.json")
    assert report["training"]["completed_updates"] == 2
    assert report["training"]["presentations"] == report["training"]["unique_examples_seen"] == 8
    assert all(row["microbatch_sizes"] == [2, 2] and row["microbatch_count"] == 2 for row in curve)
    assert all(row["padded_tokens"] == 4 * row["global_padded_width"] for row in curve)
    assert report["training"]["weights_changed"]


@pytest.mark.parametrize("microbatch", [0, -1, 5, True, 1.5])
def test_invalid_configured_microbatch_rejected(microbatch):
    with pytest.raises(ValueError, match="microbatch"):
        task.validate_config({**configuration(), "training_microbatch_size": microbatch})


@pytest.mark.parametrize("guard,updates,reason", [
    ("time", 1, "time_budget_after_completed_update"),
    ("rss", 0, "rss_budget_before_next_update"),
])
def test_training_guards_report_partial_or_zero_updates_without_crashing(tmp_path, monkeypatch, guard, updates, reason):
    output = mocked_run(tmp_path, monkeypatch, guard=guard)
    training = task.read(output / "report.json")["training"]
    assert training["completed_updates"] == updates and training["stop_reason"] == reason
    assert training["unique_examples_seen"] == training["presentations"] == updates * 4
    assert training["weights_changed"] is (updates > 0)


def test_policy_mutation_is_rejected_even_when_report_checksum_is_rebound(completed):
    path = completed / "policy.json"
    policy = task.read(path)
    policy["checkpoint"] = "final"
    task.write_json(path, policy)
    with pytest.raises(ValueError, match="checksum"):
        task.load_policy(completed)
    report = task.read(completed / "report.json")
    report["artifact_sha256"]["policy.json"] = task.digest(path)
    task.write_json(completed / "report.json", report)
    with pytest.raises(ValueError, match="validation selection"):
        task.load_policy(completed)


def test_policy_checks_implementation_and_checkpoint_manifest(completed, monkeypatch):
    policy, _ = task.load_policy(completed)
    path = completed / "initial_reader" / "reader_manifest.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="manifest differs"):
        task.load_checkpoint(completed, policy, "initial")
    monkeypatch.setattr(task, "code_digest", lambda path: "changed")
    with pytest.raises(ValueError, match="Implementation"):
        task.load_policy(completed)


def test_evaluation_length_sort_restores_source_order(monkeypatch):
    rows = fixture_rows("fixture")
    for row, size in zip(rows, (20, 10, 30, 10), strict=True):
        row["reader_tokens"] = size
    batches = []
    def predict(model, tokenizer, batch, *args):
        batches.append([row["id"] for row in batch])
        return [{"span_prediction": row["id"]} for row in batch]
    monkeypatch.setattr(task.reader, "predict_batch", predict)
    config = {**configuration(), "reader_eval_batch_size": 2}
    local, _ = task.reader_evaluation(None, None, rows, config, "fixture")
    assert batches == [["fixture-1", "fixture-3"], ["fixture-0", "fixture-2"]]
    assert [row["id"] for row in local] == [row["id"] for row in rows]


def test_benchmark_selection_is_fixed_and_public_decode_excludes_answer_text():
    rows = fixture_rows("test", 16)
    assert task.fixed_benchmark_rows(rows, 6) == rows[:6] + rows[8:14]
    with pytest.raises(ValueError, match="Insufficient"):
        task.fixed_benchmark_rows(rows, 9)
    public = task.public_decoded([{"id": "a", "span_prediction": "Paris", "null_minus_span_score": .5}])
    assert "span_prediction" not in public[0]
    assert public[0]["span_prediction_sha256"] == task.scoring.sha256("Paris")


def test_benchmark_replays_raw_generator_hash_separately_from_trimmed_display(tmp_path, monkeypatch):
    directory, output = tmp_path / "run", tmp_path / "benchmark.json"
    directory.mkdir()
    rows = fixture_rows("test", 2)
    for row in rows:
        row.update(article_sha256="movie", context_sha256=task.scoring.sha256(row["context"]),
                   question_sha256=task.scoring.sha256(row["question"]))
    config = {**configuration(), "benchmark_per_answerability": 1, "benchmark_repetitions": 2}
    raw_hash, displayed_hash = task.scoring.sha256(" Paris \n"), task.scoring.sha256("Paris")
    source, provenance = {"fixture": "source"}, {"fixture": "data"}
    policy = {"generation_config": config, "generator_source": source, "kind": "reader",
              "diagnostic_checkpoint": "initial", "diagnostic_threshold": 0}
    report = {"dataset": provenance, "test": {
        "generator": {"records": [{"id": row["id"], "prediction_sha256": raw_hash} for row in rows]},
        "diagnostic_candidate": {"records": [{"id": row["id"], "prediction_sha256": displayed_hash} for row in rows]}}}
    task.write_json(directory / "policy.json", policy)
    task.write_json(directory / "report.json", report)
    task.write_json(directory / "initial-test-decoded.json", [
        {"id": row["id"], "span_prediction_sha256": displayed_hash, "null_minus_span_score": -1} for row in rows])
    monkeypatch.setattr(task, "load_policy", lambda path: (policy, report))
    monkeypatch.setattr(task, "setup", lambda config: None)
    monkeypatch.setattr(task.data, "load_prepared", lambda *args: ({"test": rows}, provenance))
    monkeypatch.setattr(task, "load_generator", lambda *args: (None, None, 1, source))
    monkeypatch.setattr(task, "load_checkpoint", lambda *args: (None, None))
    monkeypatch.setattr(task.generator, "answer", lambda *args: {"answer": "Paris", "prediction_sha256": raw_hash})
    monkeypatch.setattr(task.reader, "predict_batch", lambda *args:
                        [{"span_prediction": "Paris", "null_minus_span_score": -1}])
    task.benchmark(directory, tmp_path / "unused-data", tmp_path / "unused-source", "paired", output)
    result = task.read(output)
    assert result["replay_matches"] == {"generator": 4, "reader": 4}
    generator_rows = [row for row in result["records"] if row["arm"] == "generator"]
    assert all(row["prediction_sha256"] == raw_hash for row in generator_rows)
    assert all(row["displayed_answer_sha256"] == displayed_hash for row in generator_rows)
    assert raw_hash != displayed_hash
    assert result["all_paired_inputs_identical"]
    assert [row["arm"] for row in result["records"]] == [
        "generator", "reader", "reader", "generator", "reader", "generator", "generator", "reader"]
