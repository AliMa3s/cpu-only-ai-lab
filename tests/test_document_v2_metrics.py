"""Strict abstention, official-score diagnostics, and validation selection."""
import copy
import json

import pytest

from cpu_ai_lab import document_v2_metrics as task


def row(identifier="q1", *, impossible=False, length="short", references=None):
    return {"id": identifier, "is_impossible": impossible, "length_group": length,
            "context_tokens": 40 if length == "short" else 300,
            "context": "The meeting is in Paris. New York is another city.",
            "question": "Where is the meeting?",
            "references": [] if impossible else references or ["Paris"]}


def scored_set(predictions, *, missing=(False, False, True, True)):
    rows = [row(f"q{i}", impossible=flag, length="short" if i % 2 == 0 else "long") for i, flag in enumerate(missing)]
    return task.summarize([task.score_prediction(source, value) for source, value in zip(rows, predictions, strict=True)])


def test_official_normalization_multireferences_and_duplicate_token_precision():
    assert task.answer_scores("The, Eiffel Tower!", ["Paris", "Eiffel Tower"]) == (1, 1)
    assert task.answer_scores("New York", ["York City"]) == (0, 0.5)
    assert task.answer_scores("Paris Paris", ["Paris"])[1] == pytest.approx(2 / 3)
    assert task.answer_scores("", [""]) == (1, 1)
    with pytest.raises(ValueError, match="reference"):
        task.answer_scores("Paris", [])


@pytest.mark.parametrize("prediction", ["NOT_FOUND", " NOT_FOUND\n\t"])
def test_only_explicit_marker_is_successful_missing_answer(prediction):
    scored = task.score_prediction(row(impossible=True), prediction)
    assert scored["explicit_abstention"] is True
    assert scored["exact_match"] == scored["token_f1"] == 1
    assert scored["official_squad_em"] == scored["official_squad_f1"] == 1
    assert scored["output_state"] == "abstained"


@pytest.mark.parametrize("prediction", ["not_found", "NOT_FOUND.", '"NOT_FOUND"', "NOT_FOUND because it is absent"])
def test_marker_variants_are_not_silently_normalized_to_abstention(prediction):
    scored = task.score_prediction(row(impossible=True), prediction)
    assert scored["explicit_abstention"] is False
    assert scored["answered"] is True
    assert scored["exact_match"] == scored["token_f1"] == 0


@pytest.mark.parametrize("prediction", ["", " \t\n", "...!", "The"])
def test_malformed_normalized_empty_output_is_not_deliberate_abstention(prediction):
    scored = task.score_prediction(row(impossible=True), prediction)
    assert scored["malformed_output"] is True
    assert scored["explicit_abstention"] is False
    assert scored["exact_match"] == scored["token_f1"] == 0
    # Official SQuAD normalization alone rewards this on a no-answer gold row.
    assert scored["official_squad_em"] == scored["official_squad_f1"] == 1


def test_answerable_marker_is_false_abstention_and_public_record_has_no_text():
    scored = task.score_prediction(row(), "NOT_FOUND")
    assert scored["exact_match"] == scored["token_f1"] == 0
    assert scored["verbatim_span_in_passage"] is False
    assert not {"context", "question", "references", "prediction", "response"} & scored.keys()
    assert "Where is the meeting" not in json.dumps(scored)
    assert scored["prediction_sha256"] == task.sha256("NOT_FOUND")


def test_span_diagnostic_is_case_insensitive_and_not_a_quality_gate():
    assert task.score_prediction(row(), " PARIS ")["verbatim_span_in_passage"]
    wrong = task.score_prediction(row(), "New York")
    assert wrong["verbatim_span_in_passage"] is True
    assert wrong["exact_match"] == 0


def test_missing_coverage_precision_and_malformed_denominators_are_explicit():
    metric = scored_set(["Paris", "NOT_FOUND", "Rome", "..."])
    assert metric["examples"] == 4
    assert metric["exact_match"] == 0.25
    assert metric["official_squad_em"] == 0.5
    assert metric["coverage"] == 0.5
    assert metric["answer_precision_exact"] == 0.5
    assert metric["false_abstention_on_answerable_rate"] == 0.5
    assert metric["false_answer_on_missing_rate"] == 0.5
    assert metric["missing_failure_rate"] == 1
    assert metric["correct_abstention_recall"] == 0
    assert metric["malformed_output_rate"] == 0.25
    assert metric["by_length"]["short"]["examples"] == 2
    assert metric["by_stratum"]["answerable_short"]["exact_match"] == 1
    assert metric["by_stratum"]["missing_long"]["malformed_examples"] == 1
    assert metric["by_stratum"]["answerable_short"]["correct_abstention_recall"] is None


def test_all_abstain_has_zero_utility_and_cannot_pass_retention_gate():
    baseline = scored_set(["Paris", "Paris", "Paris", "Paris"])
    abstain = scored_set([task.NOT_FOUND] * 4)
    assert abstain["exact_match"] == 0.5
    assert abstain["coverage"] == 0
    assert abstain["answer_precision_exact"] is None
    assert abstain["harmonic_utility"] == 0
    gate = task.checkpoint_gate(baseline, abstain)
    assert not gate["accepted"]
    assert not gate["checks"]["nonzero_answer_coverage"]
    assert not gate["checks"]["answerable_exact_match_within_absolute_0_05"]


def test_valid_improvement_passes_and_initial_adapter_itself_is_not_an_improvement():
    baseline = scored_set(["Paris", "Rome", "Paris", "Paris"])
    improved = scored_set(["Paris", "Rome", task.NOT_FOUND, task.NOT_FOUND])
    assert task.checkpoint_gate(baseline, improved)["accepted"]
    assert not task.checkpoint_gate(baseline, baseline)["accepted"]
    assert task.checkpoint_key(improved, 0.5) == (pytest.approx(2 / 3), 0.75, 0.75, -0.5)


def test_retention_floor_applies_to_both_answerable_measures():
    baseline = scored_set(["Paris", "Rome", "Paris", "Paris"])
    candidate = scored_set(["Paris", "Rome", task.NOT_FOUND, task.NOT_FOUND])
    for name, check in (("answerable_exact_match", "answerable_exact_match_within_absolute_0_05"),
                        ("answerable_token_f1", "answerable_token_f1_within_absolute_0_05")):
        bad = {**candidate, name: baseline[name] - 0.051}
        assert not task.checkpoint_gate(baseline, bad)["checks"][check]
        boundary = {**candidate, name: baseline[name] - 0.05}
        assert task.checkpoint_gate(baseline, boundary)["checks"][check]


def test_selection_ignores_standard_score_and_any_heldout_summary():
    baseline = scored_set(["Paris", "Rome", "Paris", "Paris"])
    candidate = scored_set(["Paris", "Rome", task.NOT_FOUND, task.NOT_FOUND])
    initial_gate = task.checkpoint_gate(baseline, candidate)
    initial_key = task.checkpoint_key(candidate, 0.7)
    modified = {**candidate, "official_squad_em": 0, "official_squad_f1": 0,
                "test": {"exact_match": 0, "token_f1": 0}}
    assert task.checkpoint_gate(baseline, modified) == initial_gate
    assert task.checkpoint_key(modified, 0.7) == initial_key


def test_identity_hash_catches_different_order_or_rows_and_ignores_predictions():
    sources = [row("q0"), row("q1", impossible=True)]
    first = [task.score_prediction(source, "Paris") for source in sources]
    second = [task.score_prediction(source, task.NOT_FOUND) for source in sources]
    assert task.summarize(first)["evaluation_identity_sha256"] == task.summarize(second)["evaluation_identity_sha256"]
    with pytest.raises(ValueError, match="identical validation"):
        task.checkpoint_gate(task.summarize(first), task.summarize(list(reversed(second))))
    with pytest.raises(ValueError, match="unique"):
        task.summarize(first + first)


def test_invalid_sources_and_nonfinite_scores_are_rejected():
    invalid = row()
    invalid["context_sha256"] = "bad"
    with pytest.raises(ValueError, match="context_sha256"):
        task.score_prediction(invalid, "Paris")
    with pytest.raises(ValueError, match="marker"):
        task.score_prediction(row(references=[task.NOT_FOUND]), "Paris")
    invalid = row(impossible=True)
    invalid["references"] = ["Paris"]
    with pytest.raises(ValueError, match="no gold"):
        task.score_prediction(invalid, "Paris")
    record = task.score_prediction(row(), "Paris")
    record["token_f1"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        task.summarize([record])
    metric = scored_set(["Paris", "Rome", "Paris", "Paris"])
    with pytest.raises(ValueError, match="finite"):
        task.checkpoint_key(metric, float("nan"))
    with pytest.raises(ValueError, match="finite"):
        task.checkpoint_gate(metric, {**metric, "answerable_token_f1": None})
