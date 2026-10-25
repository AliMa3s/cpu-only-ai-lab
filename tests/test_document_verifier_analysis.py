"""Saved-score uncertainty distinguishes a diagnostic verifier from deployment."""
import copy
import importlib.util
from pathlib import Path

import pytest


source = Path(__file__).resolve().parents[1] / "scripts/analyze_document_verifier.py"
spec = importlib.util.spec_from_file_location("analyze_document_verifier", source)
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


def fixture(accepted=False, *, zero_correct=False, separate_strata=False):
    baseline, candidate = [], []
    for index in range(8):
        group = index % 4
        missing = group >= 2
        exact = int(not missing and not zero_correct)
        after_exact = int(missing or (exact and index != 0))
        identity = {"id": str(index), "article_sha256": f"article-{group if separate_strata else index // 4}",
                    "context_sha256": f"c{index}", "question_sha256": f"q{index}",
                    "is_impossible": missing, "length_group": "short" if group % 2 == 0 else "long",
                    "context_tokens": 20 if group % 2 == 0 else 250}
        baseline.append({**identity, "prediction_sha256": f"before-{index}", "exact_match": exact,
                         "token_f1": float(exact), "explicit_abstention": False})
        candidate.append({**identity, "prediction_sha256": f"after-{index}", "exact_match": after_exact,
                          "token_f1": float(after_exact), "explicit_abstention": missing})
    return {"format": "cpu-document-verifier-report-v1",
            "selection": {"accepted": accepted, "enabled": accepted, "test_used_for_selection": False},
            "test": {"baseline": {"records": baseline}, "verifier_candidate": {"records": candidate},
                     "selected": {"records": copy.deepcopy(candidate if accepted else baseline)}}}


def test_rejected_candidate_and_unchanged_selected_policy_are_both_visible():
    result = task.analyze(fixture(), "source-sha", seed=31, repetitions=200)
    candidate = result["comparisons"]["verifier_candidate"]
    assert candidate["paired_correctness"]["overall"]["candidate_only_correct"] == 4
    assert candidate["paired_correctness"]["overall"]["baseline_only_correct"] == 1
    assert candidate["exact_correct_counts"]["raw_record_retention"] == 0.75
    assert candidate["intervals"]["exact_correct_retention"]["point_estimate"] == 0.75
    selected = result["comparisons"]["selected"]
    assert selected["intervals"]["exact_match"]["percentile_95_interval"] == [0, 0]
    assert selected["intervals"]["exact_correct_retention"]["percentile_95_interval"] == [1, 1]
    assert result["source_report_sha256"] == "source-sha"


def test_joint_resampling_and_accepted_policy_are_deterministic():
    source = fixture(accepted=True)
    result = task.analyze(source, "sha", repetitions=200)
    assert result == task.analyze(source, "sha", repetitions=200)
    assert result["comparisons"]["verifier_candidate"]["intervals"] == result["comparisons"]["selected"]["intervals"]
    assert result["bootstrap"]["retained_stratum_complete_replicates"] == 200


def test_undefined_retention_is_reported_without_discarding_other_metrics():
    result = task.analyze(fixture(zero_correct=True), "sha", repetitions=40)
    assert len(result["bootstrap"]["undefined_retention_draw_indices_zero_based"]) == 40
    for comparison in result["comparisons"].values():
        assert comparison["intervals"]["exact_correct_retention"]["defined_replicates"] == 0
        assert comparison["intervals"]["exact_correct_retention"]["percentile_95_interval"] is None
        assert comparison["intervals"]["exact_match"]["defined_replicates"] == 40


def test_missing_strata_draws_are_explicitly_skipped():
    result = task.analyze(fixture(separate_strata=True), "sha", seed=7, repetitions=100)
    bootstrap = result["bootstrap"]
    skipped = bootstrap["skipped_empty_stratum_draw_indices_zero_based"]
    assert 0 < len(skipped) < 100
    assert bootstrap["retained_stratum_complete_replicates"] + len(skipped) == 100
    assert skipped == sorted(set(skipped))


def test_selected_contract_and_source_identity_are_verified():
    source = fixture()
    source["test"]["selected"] = source["test"]["verifier_candidate"]
    with pytest.raises(ValueError, match="validation-approved"):
        task.analyze(source, "sha", repetitions=10)
    source = fixture()
    source["test"]["verifier_candidate"]["records"][0]["article_sha256"] = "changed"
    with pytest.raises(ValueError, match="Paired IDs"):
        task.analyze(source, "sha", repetitions=10)
    source = fixture()
    source["selection"]["test_used_for_selection"] = True
    with pytest.raises(ValueError, match="validation-only"):
        task.analyze(source, "sha", repetitions=10)
