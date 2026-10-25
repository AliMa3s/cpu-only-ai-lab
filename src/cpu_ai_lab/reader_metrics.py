"""Strict extractive-reader scoring and validation-only abstention selection.

Retention is the intersection of generator-correct and reader-correct
answerable cases. New reader successes cannot compensate for losing an old
correct answer in this measure. This module uses only the standard library.
"""
from __future__ import annotations

import math

from . import document_v2_metrics as scoring


MIN_EXACT_RETENTION = 0.95
MIN_MISSING_RECALL = 0.40
MAX_ANSWERABLE_F1_DROP = 0.02
THRESHOLD_GRID = ("always_answer",) + tuple(index / 2 for index in range(-40, 41)) + ("always_abstain",)


def _finite_score(value):
    if type(value) not in (int, float) or not math.isfinite(value):
        raise ValueError("Null-minus-span scores must be finite numbers.")
    return float(value)


def _threshold(value):
    if value in ("always_answer", "always_abstain"):
        return value
    return _finite_score(value)


def apply_threshold(span_prediction: str, null_minus_span_score: float, threshold) -> str:
    """Withhold iff null-minus-span is strictly greater; equality answers.

    Only model outputs and the frozen threshold enter this decision. A
    malformed span remains malformed in always-answer mode; it is not silently
    converted to a successful missing-answer prediction.
    """
    if not isinstance(span_prediction, str):
        raise TypeError("The reader span prediction must be text.")
    score, cutoff = _finite_score(null_minus_span_score), _threshold(threshold)
    withhold = cutoff == "always_abstain" or (cutoff != "always_answer" and score > cutoff)
    return scoring.NOT_FOUND if withhold else span_prediction


def _validate_vectors(rows, span_predictions, scores, generator_predictions):
    if not rows or any(len(values) != len(rows) for values in (span_predictions, scores, generator_predictions)):
        raise ValueError("Nonempty rows, spans, scores and generator predictions must align.")
    if any(not isinstance(value, str) for value in (*span_predictions, *generator_predictions)):
        raise TypeError("Reader and generator predictions must be text.")
    for value in scores:
        _finite_score(value)


def _paired_retention(baseline_records, candidate_records):
    if not baseline_records or len(baseline_records) != len(candidate_records):
        raise ValueError("Retention requires aligned nonempty paired records.")
    identity = ("id", "context_sha256", "question_sha256", "is_impossible", "length_group", "context_tokens")
    old_correct = retained = new_correct = candidate_correct = 0
    for baseline, candidate in zip(baseline_records, candidate_records, strict=True):
        if any(baseline[key] != candidate[key] for key in identity):
            raise ValueError("Retention requires identical paired source identities and order.")
        if baseline["exact_match"] not in (0, 1) or candidate["exact_match"] not in (0, 1):
            raise ValueError("Exact-match correctness must be binary.")
        positive = not baseline["is_impossible"]
        old = positive and baseline["exact_match"] == 1
        current = positive and candidate["exact_match"] == 1
        old_correct += int(old)
        retained += int(old and current)
        new_correct += int(not old and current)
        candidate_correct += int(current)
    return {"baseline_exact_correct": old_correct, "retained_exact_correct": retained,
            "lost_exact_correct": old_correct - retained, "new_exact_correct": new_correct,
            "candidate_exact_correct": candidate_correct,
            "exact_correct_retention": retained / old_correct if old_correct else None,
            "definition": "Among answerable questions exactly correct for the generator, the fraction also exactly correct for this candidate; newly correct questions never offset lost correct questions."}


def _summarize(records, baseline_records):
    return {**scoring.summarize(records), "paired_retention": _paired_retention(baseline_records, records), "records": records}


def _generator_metric(rows, predictions):
    records = []
    for row, prediction in zip(rows, predictions, strict=True):
        record = scoring.score_prediction(row, prediction)
        correct = not row["is_impossible"] and record["exact_match"] == 1
        records.append({**record, "generator_prediction_sha256": record["prediction_sha256"],
                        "generator_exact_correct_answerable": correct,
                        "retained_generator_exact_correct": correct,
                        "newly_exact_correct_answerable": False})
    return _summarize(records, records)


def _evaluate(rows, spans, scores, threshold, baseline):
    cutoff = _threshold(threshold)
    records = []
    for row, span, score, original in zip(rows, spans, scores, baseline["records"], strict=True):
        answer = apply_threshold(span, score, cutoff)
        record = scoring.score_prediction(row, answer)
        previous_correct = not row["is_impossible"] and original["exact_match"] == 1
        current_correct = not row["is_impossible"] and record["exact_match"] == 1
        withheld = cutoff == "always_abstain" or (cutoff != "always_answer" and score > cutoff)
        records.append({**record, "reader_span_sha256": scoring.sha256(span),
                        "null_minus_span_score": float(score), "threshold_withheld": withheld,
                        "generator_prediction_sha256": original["prediction_sha256"],
                        "generator_exact_correct_answerable": previous_correct,
                        "retained_generator_exact_correct": previous_correct and current_correct,
                        "newly_exact_correct_answerable": not previous_correct and current_correct})
    return {**_summarize(records, baseline["records"]), "threshold": cutoff}


def evaluate_reader(rows, span_predictions, null_minus_span_scores, threshold, generator_predictions):
    """Evaluate one already-fixed reader policy against aligned raw generation."""
    _validate_vectors(rows, span_predictions, null_minus_span_scores, generator_predictions)
    baseline = _generator_metric(rows, generator_predictions)
    return _evaluate(rows, span_predictions, null_minus_span_scores, threshold, baseline)


def quality_gate(baseline, candidate):
    """Validation gate using paired correctness, never aggregate-count retention."""
    scoring._selection_values(baseline)
    scoring._selection_values(candidate)
    if (baseline["evaluation_identity_sha256"] != candidate["evaluation_identity_sha256"] or
            baseline["examples"] != candidate["examples"]):
        raise ValueError("Quality gates require the identical validation examples.")
    paired = _paired_retention(baseline["records"], candidate["records"])
    correct, retained = paired["baseline_exact_correct"], paired["retained_exact_correct"]
    checks = {"positive_baseline_exact_correct_count": correct > 0,
              "paired_exact_correct_retention_at_least_0_95": correct > 0 and retained >= MIN_EXACT_RETENTION * correct,
              "answerable_f1_drop_at_most_0_02": candidate["answerable_token_f1"] >= baseline["answerable_token_f1"] - MAX_ANSWERABLE_F1_DROP,
              "missing_recall_at_least_0_40": candidate["correct_abstention_recall"] >= MIN_MISSING_RECALL}
    return {"accepted": all(checks.values()), "checks": checks, **paired,
            "answerable_f1_change": candidate["answerable_token_f1"] - baseline["answerable_token_f1"],
            "retention_and_f1_pass": all(value for key, value in checks.items() if key != "missing_recall_at_least_0_40")}


def _compact(metric):
    return {key: value for key, value in metric.items() if key != "records"}


def select_threshold(rows, span_predictions, null_minus_span_scores, generator_predictions):
    """Select one checkpoint's threshold on validation only.

    The caller supplies validation rows and freezes this result before test.
    Initial/final checkpoint comparison is deliberately left to the experiment
    runner. A failed reader never silently replaces the retained generator.
    """
    _validate_vectors(rows, span_predictions, null_minus_span_scores, generator_predictions)
    baseline = _generator_metric(rows, generator_predictions)
    if not baseline["answerable_examples"] or not baseline["unanswerable_examples"]:
        raise ValueError("Threshold selection requires both answerability classes.")
    trials, selected, selected_metric, constrained = [], None, None, None
    selected_key = constrained_key = None
    for index, threshold in enumerate(THRESHOLD_GRID):
        metric = _evaluate(rows, span_predictions, null_minus_span_scores, threshold, baseline)
        gate = quality_gate(baseline, metric)
        trial = {"threshold": threshold, "metrics": _compact(metric), "gate": gate}
        trials.append(trial)
        key = (metric["correct_abstention_recall"], metric["answerable_token_f1"],
               metric["answerable_exact_match"], metric["coverage"], -index)
        if gate["retention_and_f1_pass"] and (constrained_key is None or key > constrained_key):
            constrained, constrained_key = trial, key
        if gate["accepted"] and (selected_key is None or key > selected_key):
            selected, selected_metric, selected_key = trial, metric, key
    accepted = selected is not None
    diagnostic = constrained if constrained is not None else trials[0]
    return {"format": "cpu-ai-lab-reader-threshold-selection-v1", "accepted": accepted,
            "selected_enabled": accepted, "selected_kind": "reader" if accepted else "generator",
            "selected_threshold": selected["threshold"] if accepted else None,
            "selected": selected_metric if accepted else baseline, "baseline": baseline,
            "selected_gate": selected["gate"] if accepted else None,
            "best_constrained_candidate": constrained, "diagnostic_candidate": diagnostic,
            "diagnostic_threshold": diagnostic["threshold"],
            "diagnostic_satisfies_quality_constraints": constrained is not None,
            "trials": trials, "validation_identity_sha256": baseline["evaluation_identity_sha256"],
            "test_used_for_selection": False,
            "criteria": {"minimum_paired_exact_correct_retention": MIN_EXACT_RETENTION,
                         "maximum_answerable_f1_drop_absolute": MAX_ANSWERABLE_F1_DROP,
                         "minimum_missing_recall": MIN_MISSING_RECALL,
                         "threshold_grid": list(THRESHOLD_GRID),
                         "suppression_rule": "null_minus_span_score > threshold; equality answers; named always_answer and always_abstain endpoints",
                         "ranking": "Highest missing recall, then answerable F1, answerable EM, coverage, then earlier grid entry.",
                         "fallback": "Raw generator output if no threshold passes all validation constraints.",
                         "diagnostic": "Best retention/F1-constrained threshold; if none exists, always_answer is reported with its failed gate. Diagnostic results never promote a rejected reader."}}
