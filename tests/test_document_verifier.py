"""Fixed-feature logistic fitting, artifact integrity and selective abstention."""
import copy
import json

import numpy as np
import pytest

from cpu_ai_lab import document_verifier as task


def confidence():
    return {"mean_logprob": -0.2, "min_logprob": -0.8, "mean_entropy": 0.4,
            "mean_margin": 0.5, "generated_tokens": 3, "ended_with_eos": True}


def fit_fixture():
    x = np.zeros((8, len(task.FEATURE_NAMES)))
    x[:, 0] = [-3, -2, -1, -0.5, 0.5, 1, 2, 3]
    x[:, 1] = 10
    return x, [0, 0, 0, 0, 1, 1, 1, 1]


def evaluation_fixture():
    rows = []
    for index in range(40):
        impossible = index >= 20
        rows.append({"id": f"q{index}", "is_impossible": impossible,
                     "length_group": "short" if index % 2 == 0 else "long",
                     "context_tokens": 100 if index % 2 == 0 else 300,
                     "context": "Paris is in France.", "question": "Which city?",
                     "references": [] if impossible else ["Paris"]})
    return rows, ["Paris"] * 40


def test_fixed_features_are_finite_and_depend_only_on_allowed_inputs():
    first = task.features("The meeting in Paris was in 1999.", "When was the 1999 meeting?", "1999", confidence())
    assert len(first) == len(task.FEATURE_NAMES) == 20
    assert np.isfinite(first).all()
    named = dict(zip(task.FEATURE_NAMES, first))
    assert named["normalized_span"] == 1
    assert named["answer_number_in_question"] == 1
    assert named["question_starts_when"] == 1
    assert named["answer_tokens_in_context_fraction"] == 1
    # Gold-like extra keys in an external confidence object are never read.
    augmented = {**confidence(), "references": ["Rome"], "is_impossible": True, "id": "other"}
    assert first == task.features("The meeting in Paris was in 1999.", "When was the 1999 meeting?", "1999", augmented)
    empty = dict(zip(task.FEATURE_NAMES, task.features("", "", "...", confidence())))
    assert empty["normalized_span"] == 0 and empty["first_span_relative_position"] == -1
    with pytest.raises(ValueError, match="finite"):
        task.features("x", "y", "z", {**confidence(), "mean_entropy": float("nan")})


def test_newton_fit_is_deterministic_and_scales_training_only():
    x, y = fit_fixture()
    model = task.fit(x, y)
    assert model == task.fit(x, y)
    assert model["training"]["final_objective"] < model["training"]["initial_objective"]
    assert model["training"]["converged"]
    assert model["training"]["completed_newton_updates"] <= 100
    assert model["scaler"]["mean"][1] == 10
    assert model["scaler"]["scale"][1] == 1
    assert model["scaler"]["fit_split"] == "train_only"
    probabilities = task.predict(model, x)
    assert max(probabilities[:4]) < 0.5 < min(probabilities[4:])
    original = copy.deepcopy(model)
    task.predict(model, np.full((2, len(task.FEATURE_NAMES)), 10000))
    assert model == original
    assert all(row["objective_after"] <= row["objective_before"] for row in model["training"]["trace"])
    assert task.predict(model, []) == []


def test_model_roundtrip_and_coefficient_or_feature_contract_tampering(tmp_path):
    x, y = fit_fixture()
    model = task.fit(x, y)
    path = tmp_path / "model.json"
    task.save_model(path, model)
    assert task.load_model(path) == model
    assert task.predict(task.load_model(path), x) == task.predict(model, x)
    bad = copy.deepcopy(model)
    bad["coefficients"][0] += 0.1
    path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match="integrity"):
        task.load_model(path)
    bad = copy.deepcopy(model)
    bad["feature_fingerprint_sha256"] = "wrong"
    with pytest.raises(ValueError, match="fingerprint"):
        task.predict(bad, x)
    with pytest.raises(FileExistsError):
        task.save_model(path, model)


def test_fit_rejects_one_class_nonfinite_or_wrong_width_inputs():
    x, y = fit_fixture()
    with pytest.raises(ValueError, match="both binary"):
        task.fit(x, [0] * 8)
    with pytest.raises(ValueError, match="width"):
        task.fit(x[:, :-1], y)
    with pytest.raises(ValueError, match="100"):
        task.fit(x, y, max_iter=101)
    x[0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        task.fit(x, y)


def test_policy_only_suppresses_and_keeps_explicit_model_marker_withheld():
    assert task.apply_policy(" Paris ", 0.8, 0.7) == " Paris "
    assert task.apply_policy(" Paris ", 0.6, 0.7) == "NOT_FOUND"
    assert task.apply_policy(" NOT_FOUND ", 1.0, 0.0) == " NOT_FOUND "
    assert task.apply_policy("Paris", 1.0, "always_abstain") == "NOT_FOUND"
    assert task.apply_policy("Paris", 0.0, None, enabled=False) == "Paris"
    assert task.apply_policy("Paris", 0.5, 0.5) == "Paris"
    with pytest.raises(ValueError, match="finite"):
        task.apply_policy("Paris", float("nan"), 0.5)


def test_validation_selects_recall_under_retention_and_f1_constraints():
    rows, predictions = evaluation_fixture()
    probabilities = [0.9] * 20 + [0.1] * 20
    selected = task.select_threshold(rows, predictions, probabilities)
    assert selected["accepted"] and selected["selected_enabled"]
    assert selected["selected_threshold"] == 0.11
    assert selected["selected"]["correct_abstention_recall"] == 1
    assert selected["selected_gate"]["exact_correct_retention"] == 1
    assert len(selected["trials"]) == 102
    assert selected["trials"][-1]["threshold"] == "always_abstain"
    assert not selected["trials"][-1]["gate"]["accepted"]
    assert not selected["test_used_for_selection"]
    assert selected["best_constrained_candidate"]["threshold"] == 0.11
    assert all("prediction" not in row for row in selected["selected"]["records"])


def test_failed_target_retains_baseline_and_reports_constrained_candidate():
    rows, predictions = evaluation_fixture()
    # Only six of twenty missing questions can be rejected without damaging good answers.
    probabilities = [0.9] * 20 + [0.1] * 6 + [0.95] * 14
    selected = task.select_threshold(rows, predictions, probabilities)
    assert not selected["accepted"] and not selected["selected_enabled"]
    assert selected["selected_threshold"] is None
    assert selected["selected"] == selected["baseline"]
    diagnostic = selected["best_constrained_candidate"]
    assert diagnostic["metrics"]["correct_abstention_recall"] == 0.3
    assert diagnostic["gate"]["retention_and_f1_pass"]
    assert not diagnostic["gate"]["accepted"]


def test_f1_floor_is_stricter_than_one_correct_loss_retention_allowance():
    rows, predictions = evaluation_fixture()
    probabilities = [0.1] + [0.9] * 19 + [0.1] * 20
    selected = task.select_threshold(rows, predictions, probabilities)
    candidate = next(row for row in selected["trials"] if row["threshold"] == 0.11)
    assert candidate["gate"]["exact_correct_retention"] == 0.95
    assert candidate["gate"]["checks"]["exact_correct_retention_at_least_0_95"]
    assert not candidate["gate"]["checks"]["answerable_f1_drop_at_most_0_02"]
    assert not selected["accepted"]


def test_threshold_selection_does_not_mutate_model_scores_or_inputs():
    rows, predictions = evaluation_fixture()
    scores = [0.9] * 20 + [0.1] * 20
    original = copy.deepcopy((rows, predictions, scores))
    first = task.select_threshold(rows, predictions, scores)
    assert (rows, predictions, scores) == original
    assert first == task.select_threshold(rows, predictions, scores)
