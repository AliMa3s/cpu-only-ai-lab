"""Paired movie-cluster uncertainty for the frozen CPU reader experiment.

Every observed question retains its ordinary weight. Resampling whole movies
preserves within-movie dependence without forcing equal passage-length strata.
Only saved public prediction hashes and scores are read; no model is imported.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random


_helper_path = Path(__file__).with_name("analyze_document_v2.py")
_spec = importlib.util.spec_from_file_location("_reader_pair_helpers", _helper_path)
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
METRICS = ("exact_match", "token_f1", "answerable_exact_match", "answerable_token_f1", "missing_correct_abstention_recall")
ARMS = ("generator", "diagnostic_candidate", "selected", "initial_fixed_zero", "final_fixed_zero")
COMPARISONS = {
    "diagnostic_vs_generator": ("generator", "diagnostic_candidate"),
    "selected_vs_generator": ("generator", "selected"),
    "adaptation_fixed_zero": ("initial_fixed_zero", "final_fixed_zero"),
}
IDENTITY = ("id", "article_sha256", "context_sha256", "question_sha256", "is_impossible", "length_group", "context_tokens")


def _validate_pair(first, second):
    if not first or len(first) != len(second) or len({row["id"] for row in first}) != len(first):
        raise ValueError("Paired analysis requires equal nonempty unique question sets.")
    for before, after in zip(first, second, strict=True):
        if any(before[key] != after[key] for key in IDENTITY):
            raise ValueError("Paired source identities, movie clusters, labels or order differ.")
        for row in (before, after):
            if (type(row["is_impossible"]) is not bool or type(row["explicit_abstention"]) is not bool
                    or row["length_group"] not in ("short", "long") or not row["article_sha256"]):
                raise ValueError("Invalid source classification or movie cluster.")
            if type(row["exact_match"]) not in (int, float) or row["exact_match"] not in (0, 1):
                raise ValueError("Exact correctness must be binary.")
            f1 = row["token_f1"]
            if type(f1) not in (int, float) or not math.isfinite(f1) or not 0 <= f1 <= 1:
                raise ValueError("Token F1 must be a finite fraction.")
            if row["is_impossible"] and (row["exact_match"] != float(row["explicit_abstention"]) or f1 != float(row["explicit_abstention"])):
                raise ValueError("Native missing scores require explicit abstention.")
            if not row["is_impossible"] and row["explicit_abstention"] and (row["exact_match"] or f1):
                raise ValueError("Withheld answerable outputs cannot receive answer credit.")


def _same_outputs(first, second):
    _validate_pair(first, second)
    return all(all(a[key] == b[key] for key in ("prediction_sha256", "exact_match", "token_f1", "explicit_abstention"))
               for a, b in zip(first, second, strict=True))


def _validate(report):
    if report.get("format") != "cpu-reader-report-v1" or report["selection"].get("test_used_for_selection") is not False:
        raise ValueError("Expected a completed reader report with validation-only selection.")
    choice, test = report["selection"], report["test"]
    if choice["kind"] not in ("reader", "generator") or choice["diagnostic_checkpoint"] not in ("initial", "final"):
        raise ValueError("Invalid frozen reader selection.")
    arms = {name: test[name]["records"] for name in ARMS[:3]}
    arms.update({name + "_fixed_zero": test[name]["fixed_zero"]["records"] for name in ("initial", "final")})
    for rows in arms.values():
        _validate_pair(arms["generator"], rows)
    diagnostic = test[choice["diagnostic_checkpoint"]]["diagnostic"]["records"]
    # If accepted, the diagnostic checkpoint uses the selected threshold. Its
    # selected metric is authoritative when threshold tie ranking differs.
    if choice["kind"] == "reader":
        if choice["checkpoint"] not in ("initial", "final") or choice["checkpoint"] != choice["diagnostic_checkpoint"]:
            raise ValueError("Accepted reader checkpoint differs from the diagnostic choice.")
        expected = test[choice["checkpoint"]]["threshold_selected"]["records"]
        diagnostic = expected
    else:
        if choice["checkpoint"] is not None or choice["threshold"] is not None:
            raise ValueError("Generator fallback cannot retain a deployed reader threshold.")
        expected = arms["generator"]
    if not _same_outputs(arms["selected"], expected):
        raise ValueError("Selected outputs differ from the validation-approved deployment.")
    if not _same_outputs(arms["diagnostic_candidate"], diagnostic):
        raise ValueError("Diagnostic outputs differ from the validation-frozen checkpoint.")
    return arms


def _arm_metrics(total, arm):
    # n, positives, missing, followed by five score totals for each model arm.
    at = 3 + ARMS.index(arm) * 5
    values = total[at:at + 5]
    denominators = (total[0], total[0], total[1], total[1], total[2])
    return {key: value / denominator if denominator else None for key, value, denominator in zip(METRICS, values, denominators, strict=True)}


def _statistics(total):
    metrics = {arm: _arm_metrics(total, arm) for arm in ARMS}
    result = {}
    for index, (name, (before, after)) in enumerate(COMPARISONS.items()):
        before_correct = total[3 + ARMS.index(before) * 5 + 2]
        retained = total[3 + len(ARMS) * 5 + index]
        result[name] = {"baseline": metrics[before], "candidate": metrics[after],
                        "deltas": {key: metrics[after][key] - metrics[before][key] if metrics[before][key] is not None else None for key in METRICS},
                        "exact_correct_retention": retained / before_correct if before_correct else None}
    return result


def analyze(report, source_sha256, *, seed=20261102, repetitions=10000):
    if type(seed) is not int or type(repetitions) is not int or repetitions < 1:
        raise ValueError("A fixed integer seed and positive repetition count are required.")
    arms = _validate(report)
    baseline = arms["generator"]
    movies = sorted({row["article_sha256"] for row in baseline})
    width = 3 + len(ARMS) * 5 + len(COMPARISONS)
    clusters = {movie: [0.0] * width for movie in movies}
    totals = [0.0] * width
    for i, row in enumerate(baseline):
        positive = not row["is_impossible"]
        values = [1, int(positive), int(not positive)]
        for arm in ARMS:
            value = arms[arm][i]
            exact, f1 = value["exact_match"], value["token_f1"]
            values.extend((exact, f1, exact * positive, f1 * positive, exact * (not positive)))
        for before, after in COMPARISONS.values():
            values.append(int(positive and arms[before][i]["exact_match"] == 1 and arms[after][i]["exact_match"] == 1))
        for column, value in enumerate(values):
            clusters[row["article_sha256"]][column] += value
            totals[column] += value
    point = _statistics(totals)
    draws = {name: {key: [] for key in METRICS + ("exact_correct_retention",)} for name in COMPARISONS}
    undefined = {"no_answerable_questions": [], "no_missing_questions": [],
                 "zero_baseline_exact_correct": {name: [] for name in COMPARISONS}}
    rng, cluster_values = random.Random(seed), [clusters[movie] for movie in movies]
    for repetition in range(repetitions):
        total = [0.0] * width
        for _ in movies:
            cluster = cluster_values[rng.randrange(len(movies))]
            total = [left + right for left, right in zip(total, cluster, strict=True)]
        current = _statistics(total)
        if not total[1]:
            undefined["no_answerable_questions"].append(repetition)
        if not total[2]:
            undefined["no_missing_questions"].append(repetition)
        for name, stats in current.items():
            for key, value in {**stats["deltas"], "exact_correct_retention": stats["exact_correct_retention"]}.items():
                if value is not None:
                    draws[name][key].append(value)
            if stats["exact_correct_retention"] is None:
                undefined["zero_baseline_exact_correct"][name].append(repetition)
    comparisons = {}
    for name, (before, after) in COMPARISONS.items():
        pairs = list(zip(arms[before], arms[after], strict=True))
        correct = sum(not a["is_impossible"] and a["exact_match"] == 1 for a, _ in pairs)
        retained = sum(not a["is_impossible"] and a["exact_match"] == b["exact_match"] == 1 for a, b in pairs)
        comparisons[name] = {"baseline": before, "candidate": after,
            "paired_correctness": {"overall": _helpers.transitions(pairs),
                                   "by_answerability": {key: _helpers.transitions([(a, b) for a, b in pairs if a["is_impossible"] == missing])
                                                         for key, missing in (("answerable", False), ("missing", True))}},
            "exact_correct_counts": {"baseline_correct_answerable": correct, "retained_baseline_correct_answerable": retained,
                                     "lost_baseline_correct_answerable": correct - retained},
            "question_weighted_point_estimates": point[name],
            "intervals": {key: {"point_estimate": point[name]["exact_correct_retention"] if key == "exact_correct_retention" else point[name]["deltas"][key],
                                "defined_replicates": len(values),
                                "percentile_95_interval": [_helpers.percentile(values, .025), _helpers.percentile(values, .975)] if values else None}
                          for key, values in draws[name].items()}}
    identity = json.dumps([[row[key] for key in IDENTITY] for row in baseline], separators=(",", ":"), ensure_ascii=False).encode()
    code_hash = lambda path: hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()
    return {"format": "cpu-reader-paired-analysis-v1", "status": "complete", "source_report_sha256": source_sha256,
            "implementation_sha256": {"analyze_reader.py": code_hash(Path(__file__)), "analyze_document_v2.py": code_hash(_helper_path)},
            "implementation_hash_encoding": "UTF-8, universal newlines normalized to LF",
            "selection": report["selection"],
            "paired_evidence": {"examples": len(baseline), "movie_clusters": len(movies),
                                "ordered_identity_sha256": hashlib.sha256(identity).hexdigest(),
                                "same_question_ids_movie_context_question_hashes_labels_lengths_and_order": True,
                                "answerable_questions": int(totals[1]), "missing_questions": int(totals[2]),
                                "length_counts": {length: sum(row["length_group"] == length for row in baseline) for length in ("short", "long")}},
            "comparisons": comparisons,
            "bootstrap": {"seed": seed, "attempted_replicates": repetitions, "clusters_per_draw": len(movies),
                          "overall_score_replicates": repetitions, "undefined_draw_indices_zero_based": undefined,
                          "method": "Jointly resample the observed number of whole movie clusters with replacement for all five arms. Each question occurrence has equal weight; there is no balancing or reweighting of answerability or passage length within bootstrap draws. Percentile 95% bounds use linear interpolation.",
                          "missing_denominators": "An absent subgroup makes only its metrics undefined. A zero baseline-correct-positive count makes only that comparison's retention undefined. Other metrics retain the draw. No draw is replaced.",
                          "retention_estimand": "Level: exact-correct answerable questions shared by both arms divided by baseline exact-correct answerable questions. Newly correct questions cannot offset lost baseline successes.",
                          "delta_estimands": "Candidate minus baseline for EM, F1 and subgroup scores; all rates and changes are fractions."},
            "limitations": ["Descriptive uncertainty on this filtered historical DuoRC sample, not production prevalence, training-seed variation or unseen-corpus performance.",
                            "Movie clusters preserve repeated-question dependence; independence between movies remains an assumption.",
                            "Reader and generator have different architectures and previous QA supervision. Initial-versus-final at the same zero threshold isolates this run's adaptation more closely, but does not establish architecture-only causality.",
                            "Test results and intervals never select a threshold or promote a validation-rejected speed candidate.",
                            "Zero change intervals for a generator fallback reflect identical outputs, not certainty about absolute quality."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("results/reader/report.json"))
    parser.add_argument("--output", type=Path, default=Path("results/reader/analysis.json"))
    parser.add_argument("--seed", type=int, default=20261102)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Existing reader analysis evidence is preserved.")
    payload = args.report.read_bytes()
    result = analyze(json.loads(payload), hashlib.sha256(payload).hexdigest(), seed=args.seed, repetitions=args.bootstrap_repetitions)
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({name: {"paired_correctness": value["paired_correctness"]["overall"], "intervals": value["intervals"]}
                      for name, value in result["comparisons"].items()}, indent=2))


if __name__ == "__main__":
    main()
