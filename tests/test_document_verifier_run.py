"""Tiny tensor and mocked pipeline checks; no pretrained model is loaded."""
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cpu_ai_lab import document_verifier as verifier
from cpu_ai_lab import document_verifier_data as data
from cpu_ai_lab import document_verifier_run as task


def logits(probabilities):
    return torch.tensor(probabilities, dtype=torch.float32).log()


def test_confidence_excludes_eos_and_padding_distributions():
    good = logits([0.0, 0.6, 0.1, 0.3])
    poison = torch.full((4,), float("nan"))
    result = task.confidence_summary([good, poison, poison, poison], [1, 2, 0, 0], 2)
    assert result["mean_logprob"] == pytest.approx(math.log(0.6))
    assert result["min_logprob"] == pytest.approx(math.log(0.6))
    assert result["mean_entropy"] == pytest.approx(-sum(p * math.log(p) for p in (0.6, 0.1, 0.3)))
    assert result["mean_margin"] == pytest.approx(0.3)
    assert result["generated_tokens"] == 2 and result["ended_with_eos"]


def test_eos_only_uses_its_distribution_but_not_later_padding():
    result = task.confidence_summary([logits([0.0, 0.1, 0.7, 0.2]), torch.full((4,), float("nan"))], [2, 0], 2)
    assert result["mean_logprob"] == pytest.approx(math.log(0.7))
    assert result["generated_tokens"] == 1 and result["ended_with_eos"]


def test_unterminated_generation_uses_every_answer_position():
    result = task.confidence_summary([logits([0.0, 0.6, 0.1, 0.3]), logits([0.0, 0.2, 0.1, 0.7])], [1, 3], 2)
    assert result["mean_logprob"] == pytest.approx((math.log(0.6) + math.log(0.7)) / 2)
    assert result["min_logprob"] == pytest.approx(math.log(0.6))
    assert result["generated_tokens"] == 2 and not result["ended_with_eos"]


@pytest.mark.parametrize("ids,scores", [([], []), ([1, 3], [torch.zeros(4)])])
def test_confidence_rejects_missing_positions(ids, scores):
    with pytest.raises(ValueError, match="Missing"):
        task.confidence_summary(scores, ids, 2)


def test_confidence_rejects_nonfinite_used_distribution():
    with pytest.raises(ValueError, match="Nonfinite"):
        task.confidence_summary([torch.full((4,), float("nan"))], [1], 2)


class Tokenizer:
    pad_token_id = 0
    eos_token_id = 2

    def encode(self, text, add_special_tokens=False):
        return [int(value) for value in text.split()]

    def decode(self, ids, skip_special_tokens=True):
        return " ".join(str(value) for value in ids if value not in (0, 2))


def test_generation_masks_left_padding_and_early_eos_without_gold_inputs(monkeypatch):
    monkeypatch.setattr(task, "render_prompt", lambda tokenizer, row: row["prompt"])
    feature_inputs, calls = [], []
    monkeypatch.setattr(verifier, "features", lambda *args: feature_inputs.append(args) or [1.0] * 20)

    class Model:
        def generate(self, **kwargs):
            calls.append(kwargs)
            assert kwargs["input_ids"].tolist() == [[0, 1], [3, 1]]
            assert kwargs["attention_mask"].tolist() == [[0, 1], [1, 1]]
            continuation = torch.tensor([[1, 2, 0], [3, 1, 2]])
            sequences = torch.cat((kwargs["input_ids"], continuation), dim=1)
            if not kwargs.get("output_scores"):
                assert "return_dict_in_generate" not in kwargs
                return sequences
            poison = torch.full((4,), float("nan"))
            return SimpleNamespace(sequences=sequences, scores=(
                torch.stack((logits([0, .6, .1, .3]), logits([0, .2, .1, .7]))),
                torch.stack((poison, logits([0, .8, .1, .1]))),
                torch.stack((poison, poison)),
            ))

    # Neither gold references nor the answerability label is available here.
    rows = [{"id": "a", "prompt": "1", "context": "ctx-a", "question": "query-a"},
            {"id": "b", "prompt": "3 1", "context": "ctx-b", "question": "query-b"}]
    scored = task.generate_batch(Model(), Tokenizer(), rows, 3)
    assert [row["prediction"] for row in scored] == ["1", "3 1"]
    assert [row["generated_tokens_including_eos"] for row in scored] == [2, 3]
    assert scored[1]["confidence"]["mean_logprob"] == pytest.approx((math.log(.7) + math.log(.8)) / 2)
    assert feature_inputs[0][:3] == ("ctx-a", "query-a", "1")
    baseline = task.generate_batch(Model(), Tokenizer(), rows, 3, with_confidence=False)
    assert len(feature_inputs) == 2
    assert [row["prediction_sha256"] for row in baseline] == [row["prediction_sha256"] for row in scored]
    assert all("confidence" not in row and "features" not in row for row in baseline)
    assert all(call["do_sample"] is False for call in calls)


def test_collect_batches_by_known_length_then_restores_source_order(monkeypatch):
    batches = []

    def generate(model, tokenizer, rows, token_limit):
        batches.append([row["id"] for row in rows])
        return [{"id": row["id"]} for row in rows]

    monkeypatch.setattr(task, "generate_batch", generate)
    rows = [{"id": str(i), "prompt_tokens": length} for i, length in enumerate([30, 10, 20, 10, 40])]
    actual, elapsed = task.collect(None, None, rows, {"generation_batch_size": 2, "generation_tokens": 3}, "fixture")
    assert batches == [["1", "3"], ["2", "0"], ["4"]]
    assert [row["id"] for row in actual] == [row["id"] for row in rows]
    assert elapsed >= 0


def fixture_rows(split, copies=1):
    return [{"id": f"{split}-{copy}-{i}", "context": "Paris is in France.", "question": "Which city?",
             "is_impossible": i >= 2, "length_group": "short" if i % 2 == 0 else "long",
             "context_tokens": 10 if i % 2 == 0 else 200, "references": [] if i >= 2 else ["Paris"],
             "prompt": "1", "prompt_tokens": 1}
            for copy in range(copies) for i in range(4)]


def test_benchmark_selection_is_fixed_by_stratum_and_source_order():
    rows = fixture_rows("test", copies=4)
    assert task.fixed_benchmark_rows(rows) == rows[:12]
    # Outcomes cannot change the predetermined subset.
    for i, row in enumerate(rows):
        row["exact_match"] = i % 2
        row["seconds"] = 100 - i
    assert task.fixed_benchmark_rows(rows) == rows[:12]
    with pytest.raises(ValueError, match="Insufficient"):
        task.fixed_benchmark_rows(rows[:11])


def test_frozen_generator_disables_every_gradient_and_uses_eval(monkeypatch):
    model = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Dropout(.5))
    monkeypatch.setattr(task.generator, "load_source", lambda *args: (model, "tokenizer", {"total": 6}, {"source": "fixture"}))
    loaded, tokenizer, parameters, evidence = task.frozen_generator({}, Path("unused"))
    assert loaded is model and tokenizer == "tokenizer"
    assert all(not parameter.requires_grad for parameter in model.parameters())
    assert all(not module.training for module in model.modules())
    assert parameters == {"total": 6} and evidence == {"source": "fixture"}


@pytest.fixture
def completed_mock_run(tmp_path, monkeypatch):
    config = task.read(Path(__file__).parents[1] / "configs/document-verifier.json")
    config.update(train_size=4, validation_size=4, test_size=4)
    splits = {name: fixture_rows(name) for name in ("train", "validation", "test")}
    source = {"split_manifest_sha256": "source-manifest"}
    provenance = {"prior_manifest_sha256": ["source-manifest"]}
    prepared, output = tmp_path / "data", tmp_path / "run"
    prepared.mkdir()
    task.write_json(prepared / "split_manifest.json", {"fixture": True})
    model = torch.nn.Linear(1, 1).requires_grad_(False)
    monkeypatch.setattr(task, "setup_cpu", lambda config: None)
    monkeypatch.setattr(data, "load_prepared", lambda *args: (splits, provenance))
    monkeypatch.setattr(task, "frozen_generator", lambda *args: (model, None, {"total": 2}, source))
    monkeypatch.setattr(task, "adapter_snapshot", lambda model: {})
    order, policy_before_test = [], []

    def collect(model, tokenizer, rows, config, name):
        order.append(name)
        if name == "test":
            # A test generation must never precede a fully saved, reloadable policy.
            policy, saved_classifier = task.load_policy(output / "policy.json")
            assert policy["selection_uses_test"] is False
            assert saved_classifier["scaler"]["fit_split"] == "train_only"
            policy_before_test.append(task.file_hash(output / "policy.json"))
        else:
            assert not (output / "policy.json").exists()
        result = []
        for row in rows:
            probability = .2 if row["is_impossible"] else .8
            confidence = {"mean_logprob": math.log(probability), "min_logprob": math.log(probability),
                          "mean_entropy": .5, "mean_margin": .2, "generated_tokens": 2, "ended_with_eos": True}
            result.append({"id": row["id"], "prediction": "Paris", "prediction_sha256": task.scoring.sha256("Paris"),
                           "confidence": confidence, "features": [probability] + [0.0] * 19})
        return result, .01

    monkeypatch.setattr(task, "collect", collect)
    task.run(config, output, prepared, Path("unused-source"))
    assert order == ["train", "validation", "test"]
    assert policy_before_test == [task.file_hash(output / "policy.json")]
    assert task.read(output / "run_status.json")["status"] == "complete"
    return output


def test_run_freezes_reloadable_policy_before_test_and_preserves_generator(completed_mock_run):
    report = task.read(completed_mock_run / "report.json")
    assert report["training"]["generator_trainable_parameters"] == 0
    assert report["cpu"]["initial_adapter_sha256"] == report["cpu"]["final_adapter_sha256"]
    assert report["selection"]["test_used_for_selection"] is False
    assert report["training"]["classifier_reload_max_probability_difference"] == 0
    assert report["test"]["selected"]["correct_abstention_recall"] == 1


@pytest.mark.parametrize("field,value,message", [
    ("threshold", .99, "validation selection"),
    ("enabled", 1, "selection contract"),
    ("source", {"split_manifest_sha256": "other"}, "configuration/source"),
    ("selection_uses_test", True, "Unsupported"),
])
def test_saved_policy_rejects_unbound_changes(completed_mock_run, field, value, message):
    path = completed_mock_run / "policy.json"
    policy = task.read(path)
    policy[field] = value
    task.write_json(path, policy)
    with pytest.raises(ValueError, match=message):
        task.load_policy(path)


@pytest.mark.parametrize("name", ["verifier", "selection", "protocol"])
def test_saved_policy_rejects_changed_sibling_artifacts(completed_mock_run, name):
    path = completed_mock_run / (name + ".json")
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(ValueError, match="checksum mismatch"):
        task.load_policy(completed_mock_run / "policy.json")


@pytest.mark.parametrize("change,message", [("prompt", "Current prompt"), ("code", "Current implementation")])
def test_saved_policy_rejects_changed_prompt_or_implementation(completed_mock_run, monkeypatch, change, message):
    if change == "prompt":
        monkeypatch.setattr(task.generator.data, "INSTRUCTION", "Changed instruction")
    else:
        monkeypatch.setattr(task, "code_hash", lambda path: "changed-code")
    with pytest.raises(ValueError, match=message):
        task.load_policy(completed_mock_run / "policy.json")


def test_code_fingerprint_ignores_platform_line_endings(tmp_path):
    first, second = tmp_path / "lf.py", tmp_path / "crlf.py"
    first.write_bytes(b"a = 1\nb = 2\n")
    second.write_bytes(b"a = 1\r\nb = 2\r\n")
    assert task.code_hash(first) == task.code_hash(second)
    assert task.file_hash(first) != task.file_hash(second)
