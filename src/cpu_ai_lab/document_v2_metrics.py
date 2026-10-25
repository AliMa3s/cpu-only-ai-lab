"""Strict answer/abstention metrics for the filtered SQuAD2 CPU experiment.

Only an exact, whitespace-trimmed NOT_FOUND is deliberate abstention. Standard
SQuAD normalization is also reported separately; it can reward malformed empty
outputs on missing-answer rows and therefore never selects our checkpoint.
This module uses only the standard library and publishes no answer text.
"""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import re
import string


NOT_FOUND = "NOT_FOUND"
STRATA = ("answerable_short", "answerable_long", "missing_short", "missing_long")


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalize_answer(text: str) -> str:
    """SQuAD's lower/punctuation/article/whitespace normalization."""
    cleaned = "".join(char for char in text.lower() if char not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", cleaned).split())


def answer_scores(prediction: str, references: list[str]) -> tuple[float, float]:
    """Best-reference normalized EM and token overlap F1, including empty gold."""
    if not references:
        raise ValueError("At least one scoring reference is required.")
    predicted = normalize_answer(prediction)
    predicted_tokens = predicted.split()
    exact, f1 = [], []
    for reference in references:
        expected = normalize_answer(reference)
        expected_tokens = expected.split()
        exact.append(float(predicted == expected))
        if not predicted_tokens or not expected_tokens:
            f1.append(float(predicted_tokens == expected_tokens))
            continue
        overlap = sum((Counter(predicted_tokens) & Counter(expected_tokens)).values())
        if not overlap:
            f1.append(0.0)
            continue
        precision = overlap / len(predicted_tokens)
        recall = overlap / len(expected_tokens)
        f1.append(2 * precision * recall / (precision + recall))
    return max(exact), max(f1)


def score_prediction(row: dict, prediction: str) -> dict:
    """Score one generation and return identifiers, hashes, flags and numbers."""
    if not isinstance(prediction, str):
        raise TypeError("Prediction must be text.")
    if type(row["is_impossible"]) is not bool or row["length_group"] not in ("short", "long"):
        raise ValueError("Rows need a native boolean answerability label and short/long group.")
    if type(row["context_tokens"]) is not int or row["context_tokens"] < 1:
        raise ValueError("Context token count must be positive.")
    impossible = row["is_impossible"]
    references = row["references"]
    if not isinstance(references, list) or any(not isinstance(value, str) for value in references):
        raise ValueError("References must be a list of strings.")
    if impossible and references:
        raise ValueError("Missing-answer rows must have no gold answer references.")
    if not impossible and (not references or any(not normalize_answer(value) for value in references)):
        raise ValueError("Answerable rows require nonempty normalized references.")
    if any(value.strip() == NOT_FOUND for value in references):
        raise ValueError("The abstention marker cannot also be a gold answer.")
    for field, text in (("context_sha256", row["context"]), ("question_sha256", row.get("question", ""))):
        if field in row and row[field] != sha256(text):
            raise ValueError(f"Source {field} does not match the supplied text.")

    stripped = prediction.strip()
    abstained = stripped == NOT_FOUND
    malformed = not abstained and not normalize_answer(stripped)
    answered = not abstained and not malformed
    official_prediction = "" if abstained else prediction
    official_em, official_f1 = answer_scores(official_prediction, [""] if impossible else references)
    if impossible:
        exact = f1 = float(abstained)
    elif answered:
        exact, f1 = official_em, official_f1
    else:
        exact = f1 = 0.0
    return {
        "id": row["id"], "article_sha256": row.get("article_sha256"),
        "context_sha256": sha256(row["context"]),
        "question_sha256": row.get("question_sha256", sha256(row.get("question", ""))),
        "is_impossible": impossible, "length_group": row["length_group"],
        "context_tokens": row["context_tokens"], "prediction_sha256": sha256(prediction),
        "output_state": "abstained" if abstained else ("malformed" if malformed else "answered"),
        "explicit_abstention": abstained, "malformed_output": malformed, "answered": answered,
        "verbatim_span_in_passage": answered and stripped.casefold() in row["context"].casefold(),
        "exact_match": exact, "token_f1": f1,
        "official_squad_em": official_em, "official_squad_f1": official_f1,
    }


def harmonic_mean(first: float | None, second: float | None) -> float | None:
    if first is None or second is None:
        return None
    return 2 * first * second / (first + second) if first + second else 0.0


def _ratio(numerator: float, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _aggregate(records: list[dict]) -> dict:
    count = len(records)
    answerable = [row for row in records if not row["is_impossible"]]
    missing = [row for row in records if row["is_impossible"]]
    answered = [row for row in records if row["answered"]]
    abstained = [row for row in records if row["explicit_abstention"]]
    malformed = [row for row in records if row["malformed_output"]]
    missing_abstained = sum(row["explicit_abstention"] for row in missing)
    answerable_em = _ratio(sum(row["exact_match"] for row in answerable), len(answerable))
    missing_recall = _ratio(missing_abstained, len(missing))
    metric = {
        "examples": count,
        **{key: _ratio(sum(row[key] for row in records), count)
           for key in ("exact_match", "token_f1", "official_squad_em", "official_squad_f1")},
        "answerable_examples": len(answerable), "unanswerable_examples": len(missing),
        "answerable_exact_match": answerable_em,
        "answerable_token_f1": _ratio(sum(row["token_f1"] for row in answerable), len(answerable)),
        "correct_abstention_recall": missing_recall,
        "false_answer_on_missing_rate": _ratio(sum(row["answered"] for row in missing), len(missing)),
        "missing_failure_rate": _ratio(len(missing) - missing_abstained, len(missing)),
        "false_abstention_on_answerable_rate": _ratio(sum(row["explicit_abstention"] for row in answerable), len(answerable)),
        "coverage": _ratio(len(answered), count), "answered_examples": len(answered),
        "answer_precision_exact": _ratio(sum(row["exact_match"] for row in answered), len(answered)),
        "answer_f1_among_answered": _ratio(sum(row["token_f1"] for row in answered), len(answered)),
        "abstained_examples": len(abstained),
        "abstention_precision": _ratio(missing_abstained, len(abstained)),
        "malformed_examples": len(malformed), "malformed_output_rate": _ratio(len(malformed), count),
        "verbatim_span_rate": _ratio(sum(row["verbatim_span_in_passage"] for row in records), count),
        "harmonic_utility": harmonic_mean(answerable_em, missing_recall),
    }
    return metric


def summarize(records: list[dict]) -> dict:
    """Aggregate strict and standard-normalized scores with subgroup diagnostics.

    Rates are fractions in [0,1]. Undefined subgroup denominators produce None.
    Coverage excludes both explicit abstention and malformed normalized-empty
    output. False-answer-on-missing counts valid answers only; missing-failure
    additionally counts malformed output. Precision among answers uses exact
    matches and counts answers on missing questions as incorrect.
    """
    if not records:
        raise ValueError("Cannot score an empty evaluation set.")
    identifiers = [row["id"] for row in records]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("Evaluation IDs must be unique.")
    for row in records:
        if row["length_group"] not in ("short", "long"):
            raise ValueError("Invalid length group.")
        flags = [row[key] for key in ("answered", "explicit_abstention", "malformed_output")]
        if any(type(value) is not bool for value in flags) or sum(flags) != 1:
            raise ValueError("Each prediction must have exactly one output state.")
        if any(not isinstance(row[key], (int, float)) or not math.isfinite(row[key]) or not 0 <= row[key] <= 1
               for key in ("exact_match", "token_f1", "official_squad_em", "official_squad_f1")):
            raise ValueError("Per-record scores must be finite fractions.")
    identity = [[row[key] for key in ("id", "context_sha256", "question_sha256", "is_impossible", "length_group", "context_tokens")]
                for row in records]
    metric = _aggregate(records)
    metric["evaluation_identity_sha256"] = sha256(json.dumps(identity, separators=(",", ":"), ensure_ascii=False))
    metric["by_length"] = {name: _aggregate([row for row in records if row["length_group"] == name]) for name in ("short", "long")}
    metric["by_stratum"] = {
        name: _aggregate([row for row in records if ("missing" if row["is_impossible"] else "answerable") + "_" + row["length_group"] == name])
        for name in STRATA
    }
    return metric


def _selection_values(metric: dict) -> None:
    required = ("exact_match", "token_f1", "answerable_exact_match", "answerable_token_f1", "correct_abstention_recall", "coverage")
    if any(not isinstance(metric.get(key), (int, float)) or not math.isfinite(metric[key]) or not 0 <= metric[key] <= 1 for key in required):
        raise ValueError("Checkpoint selection requires finite metrics and both answerability classes.")


def checkpoint_gate(baseline: dict, candidate: dict) -> dict:
    """Predeclared validation-only acceptance; the function takes no test data."""
    _selection_values(baseline)
    _selection_values(candidate)
    if baseline["evaluation_identity_sha256"] != candidate["evaluation_identity_sha256"] or baseline["examples"] != candidate["examples"]:
        raise ValueError("Checkpoint comparisons require identical validation examples.")
    checks = {
        "strict_exact_match_improves": candidate["exact_match"] > baseline["exact_match"],
        "strict_token_f1_improves": candidate["token_f1"] > baseline["token_f1"],
        "answerable_exact_match_within_absolute_0_05": candidate["answerable_exact_match"] >= baseline["answerable_exact_match"] - 0.05,
        "answerable_token_f1_within_absolute_0_05": candidate["answerable_token_f1"] >= baseline["answerable_token_f1"] - 0.05,
        "explicit_abstention_recall_improves": candidate["correct_abstention_recall"] > baseline["correct_abstention_recall"],
        "nonzero_answer_coverage": candidate["coverage"] > 0,
    }
    return {"accepted": all(checks.values()), "checks": checks,
            "policy": "Strict validation EM, F1 and explicit missing-answer recall must improve; answerable EM and F1 may fall by at most 0.05 absolute; valid-answer coverage must remain positive. Initial adapter is fallback. Test data and standard-normalized scores do not select a checkpoint."}


def checkpoint_key(metrics: dict, loss: float) -> tuple[float, float, float, float]:
    """Rank accepted candidates by answer/abstention utility, then F1/EM/loss."""
    _selection_values(metrics)
    if not isinstance(loss, (int, float)) or not math.isfinite(loss) or loss < 0:
        raise ValueError("Checkpoint loss must be finite and nonnegative.")
    return (harmonic_mean(metrics["answerable_exact_match"], metrics["correct_abstention_recall"]),
            metrics["token_f1"], metrics["exact_match"], -loss)
