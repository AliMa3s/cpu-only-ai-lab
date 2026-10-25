"""Paired, clustered uncertainty uses saved numbers only; no model imports."""
import copy
import importlib.util
from pathlib import Path

import pytest


source = Path(__file__).resolve().parents[1] / "scripts/analyze_document_v2.py"
spec = importlib.util.spec_from_file_location("analyze_document_v2", source)
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


def fixture(*, identical=False, separate_strata=False):
    before, after = [], []
    changes = [(1, 0), (0, 1), (0, 1), (1, 1), (1, 1), (0, 0), (0, 0), (1, 0)]
    for index, (first, second) in enumerate(changes):
        group = index % 4
        if identical:
            second = first
        missing = group >= 2
        identity = {"id": f"q{index}", "article_sha256": f"article-{group if separate_strata else index // 4}",
                    "context_sha256": f"context-{index}", "question_sha256": f"question-{index}",
                    "is_impossible": missing, "length_group": "short" if group % 2 == 0 else "long",
                    "context_tokens": 100 if group % 2 == 0 else 300}
        before.append({**identity, "exact_match": first, "token_f1": float(first), "explicit_abstention": bool(missing and first)})
        after.append({**identity, "exact_match": second, "token_f1": float(second), "explicit_abstention": bool(missing and second)})
    return {"format": "cpu-ai-lab-document-v2-v1", "before": {"test": {"generation": {"records": before}}},
            "after": {"test": {"generation": {"records": after}}}}


def test_paired_transitions_capture_improvements_and_regressions():
    result = task.analyze(fixture(), "source-hash", seed=23, repetitions=200)
    overall = result["paired_correctness"]["overall"]
    assert overall == {"examples": 8, "both_correct": 2, "baseline_only_correct": 2,
                       "candidate_only_correct": 2, "both_incorrect": 2,
                       "net_additional_correct": 0, "candidate_minus_baseline_exact_match": 0}
    assert result["paired_correctness"]["by_stratum"]["answerable_short"]["baseline_only_correct"] == 1
    assert result["paired_correctness"]["by_stratum"]["answerable_long"]["candidate_only_correct"] == 1
    assert result["bootstrap"]["delta_intervals"]["exact_match"]["percentile_95_interval"] == [-0.25, 0.25]
    assert result["source_metrics_sha256"] == "source-hash"


def test_seeded_analysis_is_deterministic_and_identical_scores_have_zero_intervals():
    source = fixture(identical=True)
    first = task.analyze(source, "sha", seed=41, repetitions=100)
    assert first == task.analyze(source, "sha", seed=41, repetitions=100)
    assert first["bootstrap"]["retained_replicates"] == 100
    assert first["bootstrap"]["skipped_empty_stratum_replicates"] == 0
    for metric in first["bootstrap"]["delta_intervals"].values():
        assert metric["percentile_95_interval"] == [0, 0]
        assert metric["point_estimate"] == metric["replicate_mean"] == 0


def test_empty_stratum_draws_are_counted_retained_and_not_replaced():
    source = fixture(identical=True, separate_strata=True)
    first = task.analyze(source, "sha", seed=7, repetitions=100)
    second = task.analyze(source, "sha", seed=7, repetitions=100)
    boot = first["bootstrap"]
    assert first == second
    assert 0 < boot["retained_replicates"] < 100
    assert boot["retained_replicates"] + boot["skipped_empty_stratum_replicates"] == 100
    assert len(boot["skipped_draw_indices_zero_based"]) == boot["skipped_empty_stratum_replicates"]
    assert boot["skipped_draw_indices_zero_based"] == sorted(set(boot["skipped_draw_indices_zero_based"]))
    assert boot["attempted_replicates"] == 100


@pytest.mark.parametrize("field,value", [("id", "different"), ("article_sha256", "different"),
                                        ("context_sha256", "different"), ("question_sha256", "different"),
                                        ("is_impossible", True), ("context_tokens", 101), ("length_group", "long")])
def test_pairing_rejects_changed_source_identity(field, value):
    source = fixture()
    source["after"]["test"]["generation"]["records"][0][field] = value
    with pytest.raises(ValueError, match="Paired IDs"):
        task.analyze(source, "sha", repetitions=10)


def test_pairing_rejects_order_duplicate_ids_and_nonfinite_or_inconsistent_scores():
    source = fixture()
    source["after"]["test"]["generation"]["records"].reverse()
    with pytest.raises(ValueError, match="Paired IDs"):
        task.analyze(source, "sha", repetitions=10)
    source = fixture()
    for phase in ("before", "after"):
        records = source[phase]["test"]["generation"]["records"]
        records[1]["id"] = records[0]["id"]
    with pytest.raises(ValueError, match="unique"):
        task.analyze(source, "sha", repetitions=10)
    source = fixture()
    source["after"]["test"]["generation"]["records"][0]["token_f1"] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        task.analyze(source, "sha", repetitions=10)
    source = fixture()
    source["after"]["test"]["generation"]["records"][2]["explicit_abstention"] = False
    with pytest.raises(ValueError, match="explicit-abstention"):
        task.analyze(source, "sha", repetitions=10)


def test_percentiles_interpolate_and_equal_stratum_estimator_is_explicit():
    assert task.percentile([0, 1], 0.025) == 0.025
    assert task.percentile([3], 0.975) == 3
    metric = task.weighted_metrics([1, 0, 1, 0], [1, 0.5, 1, 0])
    assert metric == {"exact_match": 0.5, "token_f1": 0.625,
                      "answerable_exact_match": 0.5, "answerable_token_f1": 0.75,
                      "missing_correct_abstention_recall": 0.5}


def test_invalid_repetition_arguments_are_rejected():
    with pytest.raises(ValueError, match="positive integer"):
        task.analyze(fixture(), "sha", repetitions=0)
    with pytest.raises(ValueError, match="integer"):
        task.analyze(fixture(), "sha", seed=True, repetitions=10)


def guard_fixture(*, accepted=True):
    primary = fixture()
    baseline = primary["before"]["test"]["generation"]
    candidate = primary["after"]["test"]["generation"]
    return {"format": "cpu-ai-lab-document-span-guard-v1", "status": "complete",
            "selected_policy": "warmstart_plus_span_guard" if accepted else "unchanged_warmstart",
            "validation_gate": {"accepted": accepted}, "policy_sha256": "policy-hash",
            "test": {"warmstart": baseline, "candidate": candidate,
                     "selected": copy.deepcopy(candidate if accepted else baseline)}}


def test_guard_schema_has_its_own_source_identity_and_comparison_kind():
    result = task.analyze(guard_fixture(), "guard-report-hash", seed=23, repetitions=100)
    assert result["comparison_kind"] == "warmstart_vs_validation_selected_span_policy"
    assert result["source_report_format"] == "cpu-ai-lab-document-span-guard-v1"
    assert result["source_report_sha256"] == "guard-report-hash"
    assert result["policy_sha256"] == "policy-hash"
    assert result["selected_policy"] == "warmstart_plus_span_guard"
    assert result["paired_correctness"]["overall"]["candidate_only_correct"] == 2


def test_rejected_guard_analyzes_unchanged_selected_policy_not_candidate():
    result = task.analyze(guard_fixture(accepted=False), "guard-report-hash", repetitions=50)
    assert result["selected_policy"] == "unchanged_warmstart"
    assert result["paired_correctness"]["overall"]["candidate_only_correct"] == 0
    assert result["paired_correctness"]["overall"]["baseline_only_correct"] == 0
    for metric in result["bootstrap"]["delta_intervals"].values():
        assert metric["percentile_95_interval"] == [0, 0]


def test_guard_incomplete_or_inconsistent_selection_is_rejected():
    report = guard_fixture()
    report["status"] = "running"
    with pytest.raises(ValueError, match="completed"):
        task.analyze(report, "sha", repetitions=10)
    report = guard_fixture()
    report["validation_gate"]["accepted"] = False
    with pytest.raises(ValueError, match="validation gate"):
        task.analyze(report, "sha", repetitions=10)
    report = guard_fixture()
    report["test"]["selected"] = report["test"]["warmstart"]
    with pytest.raises(ValueError, match="gated deployment"):
        task.analyze(report, "sha", repetitions=10)
