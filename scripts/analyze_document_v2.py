"""Describe paired document-v2 predictions without additional model inference.

The bootstrap resamples whole article clusters jointly for both checkpoints.
Each retained draw gives equal weight to the four declared evaluation strata.
Only Python's standard library is used; source text and model weights are not read.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import random


STRATA = ("answerable_short", "answerable_long", "missing_short", "missing_long")
METRICS = ("exact_match", "token_f1", "answerable_exact_match",
           "answerable_token_f1", "missing_correct_abstention_recall")


def stratum(row: dict) -> str:
    return ("missing" if row["is_impossible"] else "answerable") + "_" + row["length_group"]


def _identity(row: dict) -> list:
    return [row[key] for key in ("id", "article_sha256", "context_sha256", "question_sha256",
                                "is_impossible", "length_group", "context_tokens")]


def validate_pairs(before: list[dict], after: list[dict]) -> None:
    if not before or len(before) != len(after):
        raise ValueError("Paired analysis requires equal nonempty evaluation sets.")
    if len({row["id"] for row in before}) != len(before):
        raise ValueError("Evaluation IDs must be unique.")
    for first, second in zip(before, after, strict=True):
        if _identity(first) != _identity(second):
            raise ValueError("Paired IDs, article/context/question hashes, labels, lengths or order differ.")
        for row in (first, second):
            if type(row["is_impossible"]) is not bool or row["length_group"] not in ("short", "long"):
                raise ValueError("Invalid answerability or length stratum.")
            if not isinstance(row["article_sha256"], str) or not row["article_sha256"]:
                raise ValueError("Every example requires an article cluster identifier.")
            if type(row["explicit_abstention"]) is not bool:
                raise ValueError("Explicit abstention must be a boolean.")
            if type(row["exact_match"]) not in (int, float) or row["exact_match"] not in (0, 1):
                raise ValueError("Strict exact-match scores must be binary.")
            if type(row["token_f1"]) not in (int, float) or not math.isfinite(row["token_f1"]) or not 0 <= row["token_f1"] <= 1:
                raise ValueError("Strict token F1 must be a finite fraction.")
            if row["is_impossible"] and (row["exact_match"] != float(row["explicit_abstention"])
                                          or row["token_f1"] != float(row["explicit_abstention"])):
                raise ValueError("Missing-answer strict scores must equal explicit-abstention success.")
            if not row["is_impossible"] and row["explicit_abstention"] and (row["exact_match"] or row["token_f1"]):
                raise ValueError("Abstention cannot receive answerable-question credit.")
    if {stratum(row) for row in before} != set(STRATA):
        raise ValueError("All four evaluation strata must be represented.")


def transitions(pairs: list[tuple[dict, dict]]) -> dict:
    counts = {"both_correct": 0, "baseline_only_correct": 0,
              "candidate_only_correct": 0, "both_incorrect": 0}
    for before, after in pairs:
        key = {(1, 1): "both_correct", (1, 0): "baseline_only_correct",
               (0, 1): "candidate_only_correct", (0, 0): "both_incorrect"}[
                   (int(before["exact_match"]), int(after["exact_match"]))]
        counts[key] += 1
    net = counts["candidate_only_correct"] - counts["baseline_only_correct"]
    return {"examples": len(pairs), **counts, "net_additional_correct": net,
            "candidate_minus_baseline_exact_match": net / len(pairs) if pairs else None}


def weighted_metrics(exact_means: list[float], f1_means: list[float]) -> dict:
    """The first two strata are answerable; the last two are missing-answer."""
    return {"exact_match": sum(exact_means) / 4,
            "token_f1": sum(f1_means) / 4,
            "answerable_exact_match": sum(exact_means[:2]) / 2,
            "answerable_token_f1": sum(f1_means[:2]) / 2,
            "missing_correct_abstention_recall": sum(exact_means[2:]) / 2}


def point_metrics(rows: list[dict]) -> dict:
    grouped = [[row for row in rows if stratum(row) == name] for name in STRATA]
    return weighted_metrics([sum(row["exact_match"] for row in group) / len(group) for group in grouped],
                            [sum(row["token_f1"] for row in group) / len(group) for group in grouped])


def percentile(values: list[float], fraction: float) -> float:
    """Linear interpolation between sorted order statistics (including endpoints)."""
    if not values or not 0 <= fraction <= 1:
        raise ValueError("Percentiles require values and a fraction between zero and one.")
    ordered = sorted(values)
    location = fraction * (len(ordered) - 1)
    low = math.floor(location)
    high = math.ceil(location)
    return ordered[low] + (ordered[high] - ordered[low]) * (location - low)


def comparison_records(report: dict) -> tuple[list[dict], list[dict], dict]:
    """Use the selected deployment, never an unselected guard candidate."""
    source_format = report.get("format")
    if source_format == "cpu-ai-lab-document-v2-v1":
        return (report["before"]["test"]["generation"]["records"],
                report["after"]["test"]["generation"]["records"],
                {"comparison_kind": "warmstart_vs_validation_selected_adapter",
                 "source_report_format": source_format,
                 "selected_step": report.get("selection", {}).get("selected_step")})
    if source_format == "cpu-ai-lab-document-span-guard-v1":
        if report.get("status") != "complete":
            raise ValueError("Span-guard analysis requires a completed report.")
        selected_name = report.get("selected_policy")
        if selected_name not in ("warmstart_plus_span_guard", "unchanged_warmstart"):
            raise ValueError("Unrecognized selected span policy.")
        enabled = selected_name == "warmstart_plus_span_guard"
        if report.get("validation_gate", {}).get("accepted") is not enabled:
            raise ValueError("Selected span policy disagrees with its validation gate.")
        expected = report["test"]["candidate" if enabled else "warmstart"]
        if report["test"]["selected"] != expected:
            raise ValueError("Selected span-policy metrics differ from the gated deployment.")
        return (report["test"]["warmstart"]["records"], report["test"]["selected"]["records"],
                {"comparison_kind": "warmstart_vs_validation_selected_span_policy",
                 "source_report_format": source_format, "selected_policy": selected_name,
                 "policy_sha256": report["policy_sha256"]})
    raise ValueError("Expected document-v2 metrics or a completed span-guard report.")


def analyze(metrics: dict, source_sha256: str, *, seed: int = 20261029, repetitions: int = 10000) -> dict:
    if type(seed) is not int or type(repetitions) is not int or repetitions < 1:
        raise ValueError("Seed must be an integer and bootstrap repetitions a positive integer.")
    before, after, comparison = comparison_records(metrics)
    validate_pairs(before, after)
    pairs = list(zip(before, after, strict=True))
    groups = {name: [(a, b) for a, b in pairs if stratum(a) == name] for name in STRATA}
    cluster_ids = sorted({row["article_sha256"] for row in before})
    # Preaggregate each article's four stratum counts and paired score changes.
    # This is mathematically identical to repeating its member rows per draw.
    clusters = {name: [[0, 0.0, 0.0] for _ in STRATA] for name in cluster_ids}
    for a, b in pairs:
        summary = clusters[a["article_sha256"]][STRATA.index(stratum(a))]
        summary[0] += 1
        summary[1] += b["exact_match"] - a["exact_match"]
        summary[2] += b["token_f1"] - a["token_f1"]
    cluster_values = [clusters[name] for name in cluster_ids]
    rng = random.Random(seed)
    draws = {name: [] for name in METRICS}
    skipped = []
    for repetition in range(repetitions):
        total = [[0, 0.0, 0.0] for _ in STRATA]
        for _ in cluster_ids:
            cluster = cluster_values[rng.randrange(len(cluster_values))]
            for index, (count, exact, f1) in enumerate(cluster):
                total[index][0] += count
                total[index][1] += exact
                total[index][2] += f1
        if any(count == 0 for count, _, _ in total):
            skipped.append(repetition)
            continue
        changes = weighted_metrics([exact / count for count, exact, _ in total],
                                   [f1 / count for count, _, f1 in total])
        for name, value in changes.items():
            draws[name].append(value)
    if not draws["exact_match"]:
        raise ValueError("No bootstrap draw retained every stratum; more article coverage is required.")

    baseline, candidate = point_metrics(before), point_metrics(after)
    identity = json.dumps([_identity(row) for row in before], separators=(",", ":"), ensure_ascii=False).encode()
    return {
        "format": "cpu-ai-lab-document-v2-paired-analysis-v1", "status": "complete",
        "source_metrics_sha256": source_sha256, "source_report_sha256": source_sha256,
        **comparison,
        "paired_evidence": {"examples": len(pairs), "article_clusters": len(cluster_ids),
                            "ordered_identity_sha256": hashlib.sha256(identity).hexdigest(),
                            "stratum_counts": {name: len(group) for name, group in groups.items()},
                            "same_ids_hashes_labels_lengths_and_order": True},
        "paired_correctness": {"overall": transitions(pairs),
                               "by_stratum": {name: transitions(group) for name, group in groups.items()}},
        "balanced_point_estimates": {"baseline": baseline, "candidate": candidate,
                                     "candidate_minus_baseline": {name: candidate[name] - baseline[name] for name in METRICS}},
        "bootstrap": {
            "seed": seed, "attempted_replicates": repetitions,
            "retained_replicates": len(draws["exact_match"]), "skipped_empty_stratum_replicates": len(skipped),
            "skipped_draw_indices_zero_based": skipped, "clusters_per_draw": len(cluster_ids),
            "stratum_order": list(STRATA), "confidence_level": 0.95,
            "resampling": "Sample the observed number of article clusters with replacement; use the same cluster multiplicities for baseline and candidate. Retain all member records. Skip, retain indices for, and do not replace draws missing a stratum.",
            "estimand": "Candidate minus baseline: equal weight 1/4 to each answerability/length stratum. Answerable subgroup scores weight its two length strata 1/2 each; missing recall weights its two length strata 1/2 each.",
            "interval_method": "Percentile bounds at 0.025 and 0.975, with linear interpolation between sorted order statistics; rates and deltas are fractions, not percentage points.",
            "delta_intervals": {
                name: {"point_estimate": candidate[name] - baseline[name],
                       "percentile_95_interval": [percentile(values, 0.025), percentile(values, 0.975)],
                       "replicate_mean": sum(values) / len(values), "replicate_min": min(values), "replicate_max": max(values)}
                for name, values in draws.items()
            },
        },
        "limitations": [
            "Descriptive resampling uncertainty for the fixed filtered historical test, not uncertainty over training seeds, production traffic, or unseen corpora.",
            "Within-article examples remain together; independence between article clusters is still an assumption and related articles can share facts or templates.",
            "Only observed article clusters are resampled. Filtering, balanced sampling, possible pretraining exposure and previous topic evaluation are not repaired by bootstrap intervals.",
            "The equal-stratum estimand preserves the experiment's artificial 50/50 answerability mix; it does not estimate production prevalence.",
            "Empty-stratum draws are skipped and reported, so any nonzero skip count conditions intervals on observing every stratum.",
            "Intervals and paired counts are post-selection held-out descriptions. They never choose a checkpoint, alter a gate or trigger further tuning.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", "--report", dest="metrics", type=Path,
                        default=Path("results/document-v2-metrics.json"),
                        help="Primary document-v2 metrics or the span-guard report to compare.")
    parser.add_argument("--output", type=Path, default=Path("results/document-v2-analysis.json"))
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Existing analyses are preserved; choose a new output path.")
    payload = args.metrics.read_bytes()
    result = analyze(json.loads(payload), hashlib.sha256(payload).hexdigest(), seed=args.seed,
                     repetitions=args.bootstrap_repetitions)
    result["completed_utc"] = datetime.now(timezone.utc).isoformat()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"paired_correctness": result["paired_correctness"],
                      "delta_intervals": result["bootstrap"]["delta_intervals"],
                      "skipped_empty_draws": result["bootstrap"]["skipped_empty_stratum_replicates"]}, indent=2))


if __name__ == "__main__":
    main()
