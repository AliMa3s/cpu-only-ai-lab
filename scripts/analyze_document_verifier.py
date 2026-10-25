"""Paired article-cluster uncertainty for a frozen-generator verifier trial."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random


_spec = importlib.util.spec_from_file_location("_document_pair_helpers", Path(__file__).with_name("analyze_document_v2.py"))
_helpers = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_helpers)
STRATA = _helpers.STRATA
DELTA_NAMES = _helpers.METRICS
COMPARISONS = ("verifier_candidate", "selected")


def _weighted(total, exact_column, f1_column):
    return _helpers.weighted_metrics([group[exact_column] / group[0] for group in total],
                                     [group[f1_column] / group[0] for group in total])


def _statistics(total):
    baseline = _weighted(total, 1, 2)
    result = {}
    for name, column in (("verifier_candidate", 3), ("selected", 6)):
        candidate = _weighted(total, column, column + 1)
        denominator = sum(group[1] / group[0] for group in total[:2]) / 2
        retained = sum(group[column + 2] / group[0] for group in total[:2]) / 2
        result[name] = {"baseline": baseline, "candidate": candidate,
                        "deltas": {key: candidate[key] - baseline[key] for key in DELTA_NAMES},
                        "exact_correct_retention": retained / denominator if denominator else None}
    return result


def _validate(report):
    if report.get("format") != "cpu-document-verifier-report-v1":
        raise ValueError("Expected a completed document-verifier report.")
    selection = report["selection"]
    if type(selection.get("accepted")) is not bool or selection.get("enabled") is not selection["accepted"] or selection.get("test_used_for_selection") is not False:
        raise ValueError("Verifier report must retain the validation-only selection contract.")
    baseline = report["test"]["baseline"]["records"]
    arms = {name: report["test"][name]["records"] for name in COMPARISONS}
    for rows in arms.values():
        _helpers.validate_pairs(baseline, rows)
    expected = arms["verifier_candidate"] if selection["accepted"] else baseline
    for chosen, reference in zip(arms["selected"], expected, strict=True):
        for key in ("prediction_sha256", "exact_match", "token_f1", "explicit_abstention"):
            if chosen[key] != reference[key]:
                raise ValueError("Selected test output differs from the validation-approved deployment.")
    return baseline, arms


def analyze(report: dict, source_sha256: str, *, seed: int = 20261031, repetitions: int = 10000) -> dict:
    if type(seed) is not int or type(repetitions) is not int or repetitions < 1:
        raise ValueError("A fixed integer seed and positive repetition count are required.")
    baseline, arms = _validate(report)
    article_ids = sorted({row["article_sha256"] for row in baseline})
    # Per stratum: count, baseline EM/F1, candidate EM/F1/retained-correct,
    # selected EM/F1/retained-correct. Retention counts only baseline-correct positives.
    clusters = {article: [[0.0] * 9 for _ in STRATA] for article in article_ids}
    totals = [[0.0] * 9 for _ in STRATA]
    for index, row in enumerate(baseline):
        group = STRATA.index(_helpers.stratum(row))
        value = [1, row["exact_match"], row["token_f1"]]
        for name in COMPARISONS:
            candidate = arms[name][index]
            retained = float(not row["is_impossible"] and row["exact_match"] == 1 and candidate["exact_match"] == 1)
            value.extend((candidate["exact_match"], candidate["token_f1"], retained))
        for column, number in enumerate(value):
            clusters[row["article_sha256"]][group][column] += number
            totals[group][column] += number
    point = _statistics(totals)
    rng = random.Random(seed)
    cluster_values = [clusters[name] for name in article_ids]
    draws = {name: {key: [] for key in DELTA_NAMES + ("exact_correct_retention",)} for name in COMPARISONS}
    empty_draws, undefined_retention = [], []
    for repetition in range(repetitions):
        total = [[0.0] * 9 for _ in STRATA]
        for _ in article_ids:
            cluster = cluster_values[rng.randrange(len(cluster_values))]
            for group, values in enumerate(cluster):
                for column, value in enumerate(values):
                    total[group][column] += value
        if any(group[0] == 0 for group in total):
            empty_draws.append(repetition)
            continue
        current = _statistics(total)
        if current["selected"]["exact_correct_retention"] is None:
            undefined_retention.append(repetition)
        for name in COMPARISONS:
            for key, value in current[name]["deltas"].items():
                draws[name][key].append(value)
            retention = current[name]["exact_correct_retention"]
            if retention is not None:
                draws[name]["exact_correct_retention"].append(retention)
    if not draws["selected"]["exact_match"]:
        raise ValueError("No bootstrap draw contained all four strata.")

    comparisons = {}
    for name in COMPARISONS:
        pairs = list(zip(baseline, arms[name], strict=True))
        positive_pairs = [(a, b) for a, b in pairs if not a["is_impossible"]]
        original_correct = sum(a["exact_match"] == 1 for a, _ in positive_pairs)
        retained_correct = sum(a["exact_match"] == b["exact_match"] == 1 for a, b in positive_pairs)
        intervals = {}
        for key, values in draws[name].items():
            estimate = point[name]["exact_correct_retention"] if key == "exact_correct_retention" else point[name]["deltas"][key]
            intervals[key] = {"point_estimate": estimate, "defined_replicates": len(values),
                              "percentile_95_interval": [_helpers.percentile(values, 0.025), _helpers.percentile(values, 0.975)] if values else None,
                              "replicate_mean": sum(values) / len(values) if values else None}
        comparisons[name] = {
            "interpretation": "Diagnostic threshold fixed on validation; may be rejected for deployment." if name == "verifier_candidate" else "Actual validation-selected deployment, including unchanged baseline on rejection.",
            "paired_correctness": {"overall": _helpers.transitions(pairs),
                                   "by_stratum": {group: _helpers.transitions([(a, b) for a, b in pairs if _helpers.stratum(a) == group]) for group in STRATA}},
            "exact_correct_counts": {"baseline_correct_answerable": original_correct,
                                     "retained_baseline_correct_answerable": retained_correct,
                                     "raw_record_retention": retained_correct / original_correct if original_correct else None},
            "balanced_point_estimates": point[name], "intervals": intervals,
        }
    identity = json.dumps([_helpers._identity(row) for row in baseline], separators=(",", ":"), ensure_ascii=False).encode()
    return {
        "format": "cpu-document-verifier-paired-analysis-v1", "status": "complete",
        "source_report_sha256": source_sha256,
        "selection": report["selection"],
        "paired_evidence": {"examples": len(baseline), "article_clusters": len(article_ids),
                            "same_ids_hashes_labels_lengths_and_order": True,
                            "ordered_identity_sha256": hashlib.sha256(identity).hexdigest(),
                            "stratum_counts": {name: sum(_helpers.stratum(row) == name for row in baseline) for name in STRATA}},
        "comparisons": comparisons,
        "bootstrap": {"seed": seed, "attempted_replicates": repetitions,
                      "retained_stratum_complete_replicates": repetitions - len(empty_draws),
                      "skipped_empty_stratum_draw_indices_zero_based": empty_draws,
                      "undefined_retention_draw_indices_zero_based": undefined_retention,
                      "clusters_per_draw": len(article_ids), "stratum_order": list(STRATA),
                      "method": "Jointly resample the observed number of whole article clusters with replacement for baseline, candidate and selected outputs. Preaggregated article totals reproduce member-record repetition exactly. Give each of four strata weight1/4; answerable and missing subgroups give their two length strata weight1/2. Report percentile95% bounds with linear interpolation.",
                      "retention_estimand": "Level, not change: balanced-stratum proportion of baseline-correct answerable outputs that remain exact-correct. Compute the ratio of weighted retained-correct to weighted baseline-correct rates. Draws with zero baseline-correct denominator are undefined only for retention.",
                      "delta_estimands": "Candidate or selected minus raw baseline for EM/F1 and answerability subgroup metrics. All rates and differences are fractions.",
                      "skipping": "Draws with any missing stratum are reported and skipped for every metric; draws with undefined retention still contribute to all EM/F1/recall changes. Skipped draws are not replaced."},
        "limitations": ["Post-selection descriptive uncertainty on a filtered balanced historical test, not production prevalence, training-seed variance or unseen-corpus performance.",
                        "Article clusters preserve within-article dependence; independence across articles remains an assumption and related facts can cross articles.",
                        "Neither test counts nor intervals select a threshold or relax validation acceptance. A diagnostic candidate can have useful test results while remaining rejected.",
                        "A zero interval for an unchanged selected policy reflects identical predictions by construction, not certainty about absolute answer quality.",
                        "Retention is conditional on baseline exact-correct positives; it does not measure retention of partially correct answers, which is why answerable F1 is separately reported."],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", type=Path, default=Path("results/document-verifier/report.json"))
    parser.add_argument("--output", type=Path, default=Path("results/document-verifier/analysis.json"))
    parser.add_argument("--seed", type=int, default=20261031)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Existing analysis evidence is preserved.")
    payload = args.report.read_bytes()
    result = analyze(json.loads(payload), hashlib.sha256(payload).hexdigest(), seed=args.seed,
                     repetitions=args.bootstrap_repetitions)
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({name: {"paired_correctness": value["paired_correctness"]["overall"], "intervals": value["intervals"]}
                      for name, value in result["comparisons"].items()}, indent=2))


if __name__ == "__main__":
    main()
