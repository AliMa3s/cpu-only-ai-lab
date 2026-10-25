"""Reader thresholds retain actual old successes, not just an aggregate count."""
import copy
import json

import pytest

from cpu_ai_lab import reader_metrics as task


def rows(answerable=40, missing=40):
    return [{"id": f"q{i}", "context": "Paris and Rome are cities. New York is another city.",
             "question": "Which city?", "is_impossible": i >= answerable,
             "length_group": "short" if i % 2 == 0 else "long",
             "context_tokens": 20 if i % 2 == 0 else 220,
             "references": ["Paris"] if i < answerable else []}
            for i in range(answerable + missing)]


def test_fixed_grid_and_null_score_direction_with_answer_on_equality():
    assert len(task.THRESHOLD_GRID) == 83
    assert task.THRESHOLD_GRID[:3] == ("always_answer", -20.0, -19.5)
    assert task.THRESHOLD_GRID[-2:] == (20.0, "always_abstain")
    assert task.apply_threshold(" Paris ", 1.0, 1.0) == " Paris "
    assert task.apply_threshold("Paris", 1.0001, 1.0) == "NOT_FOUND"
    assert task.apply_threshold("Paris", -1000, "always_abstain") == "NOT_FOUND"
    assert task.apply_threshold("Paris", 1000, "always_answer") == "Paris"
    assert task.apply_threshold("...", 1, "always_answer") == "..."


@pytest.mark.parametrize("score", [float("nan"), float("inf"), float("-inf"), True, "1"])
def test_policy_rejects_invalid_scores_even_at_named_endpoints(score):
    with pytest.raises(ValueError, match="finite"):
        task.apply_threshold("Paris", score, "always_answer")


def test_new_correct_answers_cannot_hide_lost_generator_successes():
    source = rows()
    generator = ["Paris"] * 20 + ["Rome"] * 60
    # The reader gains all 20 formerly wrong positives but loses two old successes.
    spans = ["Rome"] * 2 + ["Paris"] * 38 + ["Rome"] * 40
    scores = [-1.0] * 40 + [1.0] * 40
    selected = task.select_threshold(source, spans, scores, generator)
    candidate = task.evaluate_reader(source, spans, scores, 0.0, generator)
    assert candidate["answerable_exact_match"] == .95
    assert selected["baseline"]["answerable_exact_match"] == .5
    paired = candidate["paired_retention"]
    assert paired["baseline_exact_correct"] == 20
    assert paired["retained_exact_correct"] == 18
    assert paired["new_exact_correct"] == 20
    assert paired["lost_exact_correct"] == 2
    assert paired["exact_correct_retention"] == .9
    gate = task.quality_gate(selected["baseline"], candidate)
    assert gate["checks"]["answerable_f1_drop_at_most_0_02"]
    assert gate["checks"]["missing_recall_at_least_0_40"]
    assert not gate["checks"]["paired_exact_correct_retention_at_least_0_95"]
    assert not selected["accepted"] and selected["selected"] == selected["baseline"]
    assert selected["selected_kind"] == "generator"
    assert selected["best_constrained_candidate"] is None
    assert selected["diagnostic_threshold"] == "always_answer"
    assert not selected["diagnostic_candidate"]["gate"]["accepted"]


def test_one_of_twenty_old_successes_can_be_lost_only_if_f1_also_passes():
    source = rows()
    generator = ["Paris"] * 20 + ["Rome"] * 60
    spans = ["Rome"] + ["Paris"] * 39 + ["Rome"] * 40
    selected = task.select_threshold(source, spans, [-1.0] * 40 + [1.0] * 40, generator)
    assert selected["accepted"]
    assert selected["selected_threshold"] == -1.0  # equality keeps every answerable span
    assert selected["selected_gate"]["exact_correct_retention"] == .95
    assert selected["selected"]["paired_retention"]["new_exact_correct"] == 20
    assert selected["selected"]["correct_abstention_recall"] == 1


def test_partial_answer_loss_is_a_separate_gate_from_exact_retention():
    source = rows(answerable=20, missing=20)
    for row in source[:20]:
        row["references"] = ["New York"]
    generator = ["New York"] * 10 + ["York"] * 10 + ["Rome"] * 20
    spans = ["New York"] * 10 + ["Rome"] * 30
    selected = task.select_threshold(source, spans, [-1.0] * 20 + [1.0] * 20, generator)
    candidate = task.evaluate_reader(source, spans, [-1.0] * 20 + [1.0] * 20, 0, generator)
    gate = task.quality_gate(selected["baseline"], candidate)
    assert gate["exact_correct_retention"] == 1
    assert not gate["checks"]["answerable_f1_drop_at_most_0_02"]
    assert not selected["accepted"]


def test_failed_recall_retains_raw_generator_and_exposes_fixed_diagnostic():
    source = rows(answerable=20, missing=20)
    raw = ["Paris"] * 40
    # Six missing cases can be withheld while retaining all correct positives.
    scores = [-1.0] * 20 + [1.0] * 6 + [-2.0] * 14
    selected = task.select_threshold(source, raw, scores, raw)
    assert not selected["accepted"] and not selected["selected_enabled"]
    assert selected["selected_threshold"] is None
    assert selected["selected"] == selected["baseline"]
    assert selected["diagnostic_threshold"] == -1.0
    assert selected["diagnostic_satisfies_quality_constraints"]
    assert selected["diagnostic_candidate"]["metrics"]["correct_abstention_recall"] == .3
    assert not selected["diagnostic_candidate"]["gate"]["accepted"]


def test_zero_correct_generator_denominator_cannot_pass_even_for_perfect_reader():
    source = rows(answerable=4, missing=4)
    selected = task.select_threshold(source, ["Paris"] * 8, [-1.0] * 4 + [1.0] * 4, ["Rome"] * 8)
    assert not selected["accepted"]
    assert selected["baseline"]["paired_retention"]["exact_correct_retention"] is None
    assert not any(trial["gate"]["checks"]["positive_baseline_exact_correct_count"] for trial in selected["trials"])


def test_public_records_hash_text_and_mark_real_paired_transitions():
    source = rows(answerable=2, missing=2)
    candidate = task.evaluate_reader(source, ["Paris", "Paris", "Rome", "..."], [-1, -1, 1, -1], 0,
                                     ["Paris", "Rome", "Rome", "Rome"])
    first, new, withheld, malformed = candidate["records"]
    assert first["retained_generator_exact_correct"]
    assert new["newly_exact_correct_answerable"] and not new["retained_generator_exact_correct"]
    assert withheld["threshold_withheld"] and withheld["explicit_abstention"]
    assert malformed["malformed_output"] and malformed["exact_match"] == 0
    assert malformed["official_squad_em"] == 1
    assert candidate["paired_retention"]["new_exact_correct"] == 1
    payload = json.dumps(candidate)
    assert "Which city?" not in payload and "Paris" not in payload and "Rome" not in payload
    assert all(not {"context", "question", "references", "prediction"} & record.keys() for record in candidate["records"])


def test_thresholds_do_not_mutate_inputs_and_selection_is_deterministic():
    source = rows(answerable=4, missing=4)
    spans, scores, generator = ["Paris"] * 8, [-1.0] * 4 + [1.0] * 4, ["Paris"] * 8
    original = copy.deepcopy((source, spans, scores, generator))
    first = task.select_threshold(source, spans, scores, generator)
    assert first == task.select_threshold(source, spans, scores, generator)
    assert (source, spans, scores, generator) == original
    assert not first["test_used_for_selection"]
    always_abstain = first["trials"][-1]
    assert always_abstain["metrics"]["exact_match"] == .5
    assert not always_abstain["gate"]["accepted"]


def test_gate_rejects_identity_or_order_mismatch_and_recomputes_intersection():
    source = rows(answerable=4, missing=4)
    spans = ["Paris"] * 8
    chosen = task.select_threshold(source, spans, [-1] * 4 + [1] * 4, spans)
    bad = copy.deepcopy(chosen["selected"])
    bad["records"][0], bad["records"][1] = bad["records"][1], bad["records"][0]
    with pytest.raises(ValueError, match="identities"):
        task.quality_gate(chosen["baseline"], bad)
    bad = copy.deepcopy(chosen["selected"])
    bad["records"][0]["exact_match"] = 0
    # A stale top-level paired-retention summary is not trusted by the gate.
    gate = task.quality_gate(chosen["baseline"], bad)
    assert gate["exact_correct_retention"] == .75 and not gate["accepted"]
    bad["evaluation_identity_sha256"] = "other"
    with pytest.raises(ValueError, match="identical"):
        task.quality_gate(chosen["baseline"], bad)


def test_invalid_lengths_classes_ids_and_thresholds_fail_loudly():
    source = rows(answerable=2, missing=2)
    with pytest.raises(ValueError, match="align"):
        task.select_threshold(source, ["Paris"], [0] * 4, ["Paris"] * 4)
    with pytest.raises(ValueError, match="both answerability"):
        task.select_threshold(source[:2], ["Paris"] * 2, [0] * 2, ["Paris"] * 2)
    source[1]["id"] = source[0]["id"]
    with pytest.raises(ValueError, match="unique"):
        task.evaluate_reader(source, ["Paris"] * 4, [0] * 4, 0, ["Paris"] * 4)
    with pytest.raises(ValueError, match="finite"):
        task.apply_threshold("Paris", 0, "choose_best_from_test")
