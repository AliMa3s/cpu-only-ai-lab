"""One fixed lexical/confidence logistic verifier, fitted on TRAIN only.

Feature extraction reads context, question, model output and generation scores;
it never receives references or answerability labels. Model probabilities are
decision scores, not a claim of calibrated correctness. No model inference or
Torch dependency is used here.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np

from . import document_v2_metrics as scoring


FEATURE_VERSION = 1
FEATURE_NAMES = (
    "mean_logprob", "min_logprob", "mean_entropy", "mean_margin", "generated_tokens", "ended_with_eos",
    "normalized_span", "answer_word_count", "answer_character_count", "answer_unique_token_ratio",
    "answer_tokens_in_context_fraction", "answer_tokens_in_question_fraction", "question_tokens_in_context_fraction",
    "context_word_count", "question_word_count", "first_span_relative_position", "answer_has_number",
    "answer_number_in_question", "question_starts_who", "question_starts_when",
)
FEATURE_CONTRACT = {
    "version": FEATURE_VERSION, "names": list(FEATURE_NAMES),
    "confidence": "First six fields are supplied generation statistics, in the listed order; ended_with_eos is a boolean. They are not calibrated probabilities.",
    "normalization": "SQuAD lowercase, ASCII punctuation removal, English article removal, and whitespace collapse.",
    "lexical": "Lengths count normalized characters/whitespace words. Overlap fractions use unique token sets and denominator zero gives zero. Span is a nonempty normalized literal substring; first position divides by normalized context character length, with -1 for absent. Number features use digit sequences in normalized text. Who/when check the first normalized question word.",
    "inputs": "Only complete context, question, raw model prediction, and generation confidence; no references, labels, IDs, article names or gold-derived fields.",
}
L2 = 0.01
MAX_ITERATIONS = 100
THRESHOLD_GRID = tuple(index / 100 for index in range(101)) + ("always_abstain",)
MIN_EXACT_RETENTION = 0.95
MAX_ANSWERABLE_F1_DROP = 0.02
MIN_MISSING_RECALL = 0.40


def fingerprint(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


FEATURE_FINGERPRINT = fingerprint(FEATURE_CONTRACT)


def features(context: str, question: str, prediction: str, confidence: dict) -> list[float]:
    if any(not isinstance(value, str) for value in (context, question, prediction)):
        raise TypeError("Context, question and prediction must be text.")
    names = FEATURE_NAMES[:5]
    if any(type(confidence.get(name)) not in (int, float) or not math.isfinite(confidence[name]) for name in names):
        raise ValueError("Generation confidence must contain five finite numeric statistics.")
    if confidence["generated_tokens"] < 0 or not float(confidence["generated_tokens"]).is_integer():
        raise ValueError("Generated token count must be a nonnegative integer.")
    if type(confidence.get("ended_with_eos")) is not bool:
        raise ValueError("ended_with_eos must be a boolean.")
    passage, query, answer = [scoring.normalize_answer(value) for value in (context, question, prediction)]
    context_words, question_words, answer_words = passage.split(), query.split(), answer.split()
    context_set, question_set, answer_set = set(context_words), set(question_words), set(answer_words)
    overlap = lambda first, second: len(first & second) / len(first) if first else 0.0
    position = passage.find(answer) if answer else -1
    answer_numbers, question_numbers = set(re.findall(r"\d+", answer)), set(re.findall(r"\d+", query))
    values = [float(confidence[name]) for name in names] + [float(confidence["ended_with_eos"])]
    values += [float(position >= 0), float(len(answer_words)), float(len(answer)),
               len(answer_set) / len(answer_words) if answer_words else 0.0,
               overlap(answer_set, context_set), overlap(answer_set, question_set), overlap(question_set, context_set),
               float(len(context_words)), float(len(question_words)), position / len(passage) if position >= 0 and passage else -1.0,
               float(bool(answer_numbers)), float(bool(answer_numbers & question_numbers)),
               float(bool(question_words) and question_words[0] == "who"),
               float(bool(question_words) and question_words[0] == "when")]
    if len(values) != len(FEATURE_NAMES) or not all(math.isfinite(value) for value in values):
        raise ValueError("Feature vector must contain exactly the fixed finite features.")
    return values


def _matrix(values, *, allow_empty=False):
    array = np.asarray(values, dtype=np.float64)
    if allow_empty and array.size == 0:
        return np.empty((0, len(FEATURE_NAMES)), dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != len(FEATURE_NAMES) or not array.shape[0] or not np.isfinite(array).all():
        raise ValueError("Feature matrix must be finite, nonempty, and have the declared feature width.")
    return array


def _sigmoid(values):
    return np.exp(-np.logaddexp(0.0, -values))


def _objective(design, labels, weights, l2):
    logits = design @ weights
    mean_loss = float(np.mean(np.logaddexp(0.0, logits) - labels * logits))
    penalty = 0.5 * l2 * float(weights[1:] @ weights[1:])
    return mean_loss + penalty, mean_loss


def _model_payload(model):
    return {key: value for key, value in model.items() if key != "model_integrity_sha256"}


def validate_model(model: dict) -> dict:
    if model.get("format") != "cpu-ai-lab-logistic-answer-verifier-v1":
        raise ValueError("Unsupported verifier model format.")
    if model.get("feature_contract") != FEATURE_CONTRACT or model.get("feature_fingerprint_sha256") != FEATURE_FINGERPRINT:
        raise ValueError("Verifier feature version or fingerprint differs.")
    if model.get("model_integrity_sha256") != fingerprint(_model_payload(model)):
        raise ValueError("Verifier model integrity checksum differs.")
    mean, scale = np.asarray(model["scaler"]["mean"], dtype=np.float64), np.asarray(model["scaler"]["scale"], dtype=np.float64)
    coefficients = np.asarray(model["coefficients"], dtype=np.float64)
    if any(value.shape != (len(FEATURE_NAMES),) for value in (mean, scale, coefficients)):
        raise ValueError("Verifier coefficient/scaler width differs from its feature contract.")
    if not all(np.isfinite(value).all() for value in (mean, scale, coefficients)) or np.any(scale <= 0):
        raise ValueError("Verifier scaler and coefficients must be finite with positive scales.")
    if type(model["intercept"]) not in (int, float) or not math.isfinite(model["intercept"]):
        raise ValueError("Verifier intercept must be finite.")
    return model


def fit(x, y, l2: float = L2, max_iter: int = MAX_ITERATIONS) -> dict:
    """Deterministic float64 Newton/Armijo fit; intercept is not regularized.

    One training fit is performed. No validation or test values participate in
    scaling, objective optimization, regularization choice or stopping rules.
    """
    values = _matrix(x)
    labels = np.asarray(y, dtype=np.float64)
    if labels.shape != (len(values),) or not np.isfinite(labels).all() or set(labels.tolist()) != {0.0, 1.0}:
        raise ValueError("Training labels must align with rows and contain both binary classes.")
    if type(l2) not in (int, float) or not math.isfinite(l2) or l2 <= 0:
        raise ValueError("L2 regularization must be finite and positive.")
    if type(max_iter) is not int or not 1 <= max_iter <= MAX_ITERATIONS:
        raise ValueError("Newton fitting permits at most 100 positive iterations.")
    mean = values.mean(axis=0)
    scale = values.std(axis=0, ddof=0)
    constant = scale <= 1e-12
    scale[constant] = 1.0
    standardized = (values - mean) / scale
    if not np.isfinite(standardized).all():
        raise ValueError("Training feature standardization produced nonfinite values.")
    design = np.column_stack((np.ones(len(values)), standardized))
    weights = np.zeros(design.shape[1], dtype=np.float64)
    regularization = np.full(len(weights), l2, dtype=np.float64)
    regularization[0] = 0.0
    initial, _ = _objective(design, labels, weights, l2)
    history, converged, reason = [], False, "iteration_limit"
    for iteration in range(1, max_iter + 1):
        objective, mean_loss = _objective(design, labels, weights, l2)
        probabilities = _sigmoid(design @ weights)
        gradient = design.T @ (probabilities - labels) / len(values) + regularization * weights
        gradient_norm = float(np.max(np.abs(gradient)))
        if gradient_norm <= 1e-8:
            converged, reason = True, "gradient_tolerance"
            break
        variance = probabilities * (1 - probabilities)
        hessian = (design.T * variance) @ design / len(values) + np.diag(regularization)
        # Fixed tiny diagonal damping keeps redundant/constant feature systems stable.
        direction = np.linalg.solve(hessian + np.eye(len(weights)) * 1e-10, gradient)
        slope = float(gradient @ direction)
        if not np.isfinite(direction).all() or slope <= 0:
            raise RuntimeError("Newton direction is not a finite descent direction.")
        accepted, step = False, 1.0
        for _ in range(30):
            candidate = weights - step * direction
            next_objective, next_mean_loss = _objective(design, labels, candidate, l2)
            if math.isfinite(next_objective) and next_objective <= objective - 1e-4 * step * slope:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            reason = "line_search_stopped"
            break
        history.append({"iteration": iteration, "objective_before": objective, "objective_after": next_objective,
                        "mean_log_loss_after": next_mean_loss, "step_size": step,
                        "gradient_infinity_norm_before": gradient_norm})
        weights = candidate
        if objective - next_objective <= 1e-12 * (1 + abs(objective)):
            converged, reason = True, "objective_tolerance"
            break
    final, mean_loss = _objective(design, labels, weights, l2)
    final_gradient = design.T @ (_sigmoid(design @ weights) - labels) / len(values) + regularization * weights
    model = {"format": "cpu-ai-lab-logistic-answer-verifier-v1", "feature_contract": json.loads(json.dumps(FEATURE_CONTRACT)),
             "feature_fingerprint_sha256": FEATURE_FINGERPRINT,
             "scaler": {"fit_split": "train_only", "mean": mean.tolist(), "scale": scale.tolist(),
                        "constant_feature_indices": np.flatnonzero(constant).tolist(), "standard_deviation_ddof": 0},
             "intercept": float(weights[0]), "coefficients": weights[1:].tolist(),
             "training": {"examples": len(values), "features": len(FEATURE_NAMES),
                          "positive_examples": int(labels.sum()), "negative_examples": int(len(labels) - labels.sum()),
                          "l2": l2, "intercept_regularized": False, "max_iterations": max_iter,
                          "completed_newton_updates": len(history), "converged": converged, "stop_reason": reason,
                          "initial_objective": initial, "final_objective": final, "final_mean_log_loss": mean_loss,
                          "final_gradient_infinity_norm": float(np.max(np.abs(final_gradient))), "trace": history}}
    model["model_integrity_sha256"] = fingerprint(model)
    return validate_model(model)


def predict(model: dict, x) -> list[float]:
    validate_model(model)
    values = _matrix(x, allow_empty=True)
    scaled = (values - np.asarray(model["scaler"]["mean"])) / np.asarray(model["scaler"]["scale"])
    logits = scaled @ np.asarray(model["coefficients"]) + model["intercept"]
    probabilities = _sigmoid(logits)
    if not np.isfinite(probabilities).all():
        raise ValueError("Verifier predictions are nonfinite.")
    return probabilities.tolist()


def save_model(path: Path, model: dict) -> str:
    validate_model(model)
    path = Path(path)
    payload = (json.dumps(model, indent=2, allow_nan=False) + "\n").encode()
    if path.exists() and path.read_bytes() != payload:
        raise FileExistsError("Existing verifier model bytes are preserved.")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return hashlib.sha256(payload).hexdigest()


def load_model(path: Path) -> dict:
    return validate_model(json.loads(Path(path).read_bytes()))


def _probability(value) -> float:
    if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("Verifier probabilities must be finite fractions.")
    return float(value)


def apply_policy(prediction: str, probability: float, threshold, enabled: bool = True) -> str:
    if not isinstance(prediction, str) or type(enabled) is not bool:
        raise ValueError("Policy requires text and a boolean enabled flag.")
    _probability(probability)
    if not enabled:
        return prediction
    if threshold != "always_abstain":
        _probability(threshold)
    if prediction.strip() == scoring.NOT_FOUND:
        return prediction
    return scoring.NOT_FOUND if threshold == "always_abstain" or probability < threshold else prediction


def evaluate_policy(rows, predictions, probabilities, threshold, enabled: bool = True) -> dict:
    if not rows or len(rows) != len(predictions) or len(rows) != len(probabilities):
        raise ValueError("Rows, raw predictions and probabilities must have matching nonzero lengths.")
    records = []
    for row, original, probability in zip(rows, predictions, probabilities, strict=True):
        answer = apply_policy(original, probability, threshold, enabled)
        model_abstained = original.strip() == scoring.NOT_FOUND
        suppressed = answer == scoring.NOT_FOUND and not model_abstained
        records.append({**scoring.score_prediction(row, answer),
                        "model_prediction_sha256": scoring.sha256(original),
                        "verifier_probability": float(probability), "verifier_suppressed": suppressed,
                        "abstention_origin": "model" if model_abstained else ("verifier" if suppressed else None)})
    return {**scoring.summarize(records), "suppressed_examples": sum(row["verifier_suppressed"] for row in records),
            "records": records}


def _selection_gate(baseline, candidate) -> dict:
    correct = round(baseline["answerable_exact_match"] * baseline["answerable_examples"])
    retained = round(candidate["answerable_exact_match"] * candidate["answerable_examples"])
    retention = retained / correct if correct else None
    checks = {"positive_baseline_exact_correct_count": correct > 0,
              "exact_correct_retention_at_least_0_95": correct > 0 and retained >= MIN_EXACT_RETENTION * correct,
              "answerable_f1_drop_at_most_0_02": candidate["answerable_token_f1"] >= baseline["answerable_token_f1"] - MAX_ANSWERABLE_F1_DROP,
              "missing_recall_at_least_0_40": candidate["correct_abstention_recall"] >= MIN_MISSING_RECALL}
    return {"accepted": all(checks.values()), "checks": checks,
            "baseline_exact_correct": correct, "retained_exact_correct": retained, "exact_correct_retention": retention,
            "retention_and_f1_pass": all(value for key, value in checks.items() if key != "missing_recall_at_least_0_40")}


def _compact(metric):
    return {key: value for key, value in metric.items() if key != "records"}


def select_threshold(rows, predictions, probabilities) -> dict:
    """One fixed-grid validation search; no test rows should be supplied here.

    A threshold only replaces model output by NOT_FOUND. All comparator calls
    must supply the same raw-model baseline predictions. The diagnostic best
    constrained candidate is never deployed unless all acceptance checks pass.
    """
    baseline = evaluate_policy(rows, predictions, probabilities, None, enabled=False)
    if not baseline["answerable_examples"] or not baseline["unanswerable_examples"]:
        raise ValueError("Threshold validation must contain both answerability classes.")
    trials, selected, selected_metrics, best_constrained = [], None, None, None
    best_key, constrained_key = None, None
    for index, threshold in enumerate(THRESHOLD_GRID):
        metric = evaluate_policy(rows, predictions, probabilities, threshold)
        gate = _selection_gate(baseline, metric)
        trial = {"threshold": threshold, "metrics": _compact(metric), "gate": gate}
        trials.append(trial)
        key = (metric["correct_abstention_recall"], metric["answerable_token_f1"],
               metric["answerable_exact_match"], metric["coverage"], -index)
        if gate["retention_and_f1_pass"] and (constrained_key is None or key > constrained_key):
            best_constrained, constrained_key = trial, key
        if gate["accepted"] and (best_key is None or key > best_key):
            selected, selected_metrics, best_key = trial, metric, key
    accepted = selected is not None
    return {"format": "cpu-ai-lab-verifier-threshold-selection-v1", "accepted": accepted,
            "selected_enabled": accepted, "selected_threshold": selected["threshold"] if accepted else None,
            "selected": selected_metrics if accepted else baseline, "baseline": baseline,
            "selected_gate": selected["gate"] if accepted else None,
            "best_constrained_candidate": best_constrained, "trials": trials,
            "validation_identity_sha256": baseline["evaluation_identity_sha256"],
            "test_used_for_selection": False,
            "criteria": {"minimum_exact_correct_retention": MIN_EXACT_RETENTION,
                         "maximum_answerable_f1_drop_absolute": MAX_ANSWERABLE_F1_DROP,
                         "minimum_missing_recall": MIN_MISSING_RECALL,
                         "threshold_grid": list(THRESHOLD_GRID), "suppression_rule": "score < threshold; exact model NOT_FOUND is always withheld",
                         "ranking": "Highest missing recall, then answerable F1, answerable EM, answer coverage, then earlier grid entry.",
                         "fallback": "Unchanged raw model output when no grid entry passes every validation constraint.",
                         "diagnostic": "Best constrained candidate satisfies exact-retention and F1 constraints but may miss the required missing-answer recall; it is not deployed on a failed gate."}}
