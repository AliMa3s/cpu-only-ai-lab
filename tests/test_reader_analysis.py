"""Movie resampling preserves question weighting and deployment decisions."""
import copy
import importlib.util
from pathlib import Path

import pytest


source = Path(__file__).resolve().parents[1] / "scripts/analyze_reader.py"
spec = importlib.util.spec_from_file_location("reader_analysis", source)
task = importlib.util.module_from_spec(spec)
spec.loader.exec_module(task)


def fixture(accepted=False, zero_correct=False):
    generator, candidate = [], []
    for index in range(8):
        missing = index >= 4
        before_exact = int(index in (0, 1) and not zero_correct)
        after_exact = int(index in (1, 2, 4, 5))
        identity = {"id": str(index), "article_sha256": f"movie-{index // 2}",
                    "context_sha256": f"context-{index // 2}", "question_sha256": f"question-{index}",
                    "is_impossible": missing, "length_group": "long" if index < 2 else "short",
                    "context_tokens": 250 if index < 2 else 50}
        generator.append({**identity, "exact_match": before_exact, "token_f1": float(before_exact),
                          "explicit_abstention": False, "prediction_sha256": f"g-{index}"})
        candidate.append({**identity, "exact_match": after_exact, "token_f1": float(after_exact),
                          "explicit_abstention": missing and bool(after_exact), "prediction_sha256": f"r-{index}"})
    return {"format": "cpu-reader-report-v1",
            "selection": {"kind": "reader" if accepted else "generator", "checkpoint": "final" if accepted else None,
                          "threshold": 0 if accepted else None, "diagnostic_checkpoint": "final",
                          "diagnostic_threshold": 0, "test_used_for_selection": False},
            "test": {"generator": {"records": generator}, "diagnostic_candidate": {"records": candidate},
                     "selected": {"records": copy.deepcopy(candidate if accepted else generator)},
                     "initial": {"fixed_zero": {"records": copy.deepcopy(generator)}, "diagnostic": {"records": copy.deepcopy(generator)}, "threshold_selected": None},
                     "final": {"fixed_zero": {"records": copy.deepcopy(candidate)}, "diagnostic": {"records": copy.deepcopy(candidate)},
                               "threshold_selected": {"records": copy.deepcopy(candidate)} if accepted else None}}}


def test_question_weighting_and_true_intersection_retention():
    result = task.analyze(fixture(), "source-sha", repetitions=100)
    candidate = result["comparisons"]["diagnostic_vs_generator"]
    assert candidate["intervals"]["exact_match"]["point_estimate"] == .25
    assert candidate["intervals"]["answerable_exact_match"]["point_estimate"] == 0
    assert candidate["intervals"]["exact_correct_retention"]["point_estimate"] == .5
    assert candidate["paired_correctness"]["overall"]["candidate_only_correct"] == 3
    assert candidate["paired_correctness"]["overall"]["baseline_only_correct"] == 1
    assert result["paired_evidence"]["length_counts"] == {"short": 6, "long": 2}
    assert result["source_report_sha256"] == "source-sha"
    assert len(result["implementation_sha256"]["analyze_reader.py"]) == 64


def test_rejected_speed_candidate_does_not_change_selected_output():
    result = task.analyze(fixture(), "sha", repetitions=200)
    selected = result["comparisons"]["selected_vs_generator"]
    assert selected["intervals"]["exact_match"]["percentile_95_interval"] == [0, 0]
    assert selected["intervals"]["exact_correct_retention"]["percentile_95_interval"] == [1, 1]
    assert selected["paired_correctness"]["overall"]["net_additional_correct"] == 0


def test_accepted_selected_and_diagnostic_share_joint_deterministic_draws():
    report = fixture(accepted=True)
    result = task.analyze(report, "sha", seed=61, repetitions=100)
    assert result == task.analyze(report, "sha", seed=61, repetitions=100)
    assert result["comparisons"]["diagnostic_vs_generator"]["intervals"] == result["comparisons"]["selected_vs_generator"]["intervals"]
    assert result["comparisons"]["adaptation_fixed_zero"]["intervals"] == result["comparisons"]["diagnostic_vs_generator"]["intervals"]


def test_missing_subgroups_only_skip_the_undefined_metrics():
    result = task.analyze(fixture(), "sha", seed=11, repetitions=200)
    bootstrap = result["bootstrap"]
    assert bootstrap["overall_score_replicates"] == 200
    undefined = bootstrap["undefined_draw_indices_zero_based"]
    assert undefined["no_answerable_questions"]
    assert undefined["no_missing_questions"]
    for value in result["comparisons"].values():
        assert value["intervals"]["exact_match"]["defined_replicates"] == 200
        assert value["intervals"]["answerable_token_f1"]["defined_replicates"] == 200 - len(undefined["no_answerable_questions"])
        assert value["intervals"]["missing_correct_abstention_recall"]["defined_replicates"] == 200 - len(undefined["no_missing_questions"])


def test_zero_correct_denominator_does_not_discard_other_scores():
    result = task.analyze(fixture(zero_correct=True), "sha", repetitions=100)
    candidate = result["comparisons"]["diagnostic_vs_generator"]
    assert candidate["intervals"]["exact_correct_retention"]["point_estimate"] is None
    assert candidate["intervals"]["exact_correct_retention"]["defined_replicates"] == 0
    assert candidate["intervals"]["exact_match"]["defined_replicates"] == 100


def test_all_short_sample_is_allowed_without_inventing_length_weights():
    report = fixture()
    for metric in (report["test"]["generator"], report["test"]["diagnostic_candidate"], report["test"]["selected"],
                   report["test"]["initial"]["fixed_zero"], report["test"]["initial"]["diagnostic"],
                   report["test"]["final"]["fixed_zero"], report["test"]["final"]["diagnostic"]):
        for row in metric["records"]:
            row["length_group"], row["context_tokens"] = "short", 50
    result = task.analyze(report, "sha", repetitions=100)
    assert result["paired_evidence"]["length_counts"]["long"] == 0
    assert result["comparisons"]["diagnostic_vs_generator"]["intervals"]["exact_match"]["point_estimate"] == .25


def test_report_selection_identity_and_strict_score_contracts_are_checked():
    report = fixture()
    report["test"]["selected"] = report["test"]["diagnostic_candidate"]
    with pytest.raises(ValueError, match="validation-approved"):
        task.analyze(report, "sha", repetitions=20)
    report = fixture()
    report["test"]["diagnostic_candidate"]["records"][0]["article_sha256"] = "different-movie"
    with pytest.raises(ValueError, match="source identities"):
        task.analyze(report, "sha", repetitions=20)
    report = fixture()
    report["test"]["generator"]["records"][4]["exact_match"] = 1
    with pytest.raises(ValueError, match="explicit abstention"):
        task.analyze(report, "sha", repetitions=20)
    report = fixture()
    report["selection"]["test_used_for_selection"] = True
    with pytest.raises(ValueError, match="validation-only"):
        task.analyze(report, "sha", repetitions=20)


def test_analysis_refuses_to_overwrite_evidence_before_reading_inputs(tmp_path, monkeypatch):
    output = tmp_path / "analysis.json"
    output.write_text("preserve me", encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["analyze_reader.py", "--output", str(output), "--report", str(tmp_path / "does-not-exist.json")])
    with pytest.raises(FileExistsError, match="preserved"):
        task.main()
    assert output.read_text(encoding="utf-8") == "preserve me"


def test_figure_refuses_to_overwrite_before_importing_plotting_runtime(tmp_path):
    plot_source = source.with_name("plot_reader.py")
    plot_spec = importlib.util.spec_from_file_location("reader_plot", plot_source)
    module = importlib.util.module_from_spec(plot_spec)
    plot_spec.loader.exec_module(module)
    output = tmp_path / "reader"
    output.with_suffix(".png").write_bytes(b"previous figure")
    report = fixture()
    choice = {key: value for key, value in report["selection"].items() if key != "test_used_for_selection"}
    selection = {"format": "cpu-reader-selection-v1", "uses_test": False, "choice": choice}
    with pytest.raises(FileExistsError, match="preserved"):
        module.plot(report, selection, None, output)
    assert output.with_suffix(".png").read_bytes() == b"previous figure"


def test_threshold_curve_places_named_modes_at_the_correct_numeric_limits():
    plot_spec = importlib.util.spec_from_file_location("reader_plot_order", source.with_name("plot_reader.py"))
    module = importlib.util.module_from_spec(plot_spec)
    plot_spec.loader.exec_module(module)
    trials = [{"threshold": threshold} for threshold in ("always_answer", -20, 0, 20, "always_abstain")]
    assert [row["threshold"] for row in sorted(trials, key=module._threshold_order)] == ["always_abstain", -20, 0, 20, "always_answer"]
