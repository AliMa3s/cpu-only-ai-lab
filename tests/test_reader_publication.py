"""Publication safeguards and independent audit formulas use only tiny fixtures."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
from types import SimpleNamespace

import pytest


def load_script(name):
    path = Path(__file__).resolve().parents[1] / "scripts" / (name + ".py")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_script("audit_reader")
exporter = load_script("export_reader")


def write(path, value):
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def completed_fixture(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    write(run / "run_status.json", {"status": "complete"})
    for name in exporter.ALLOWED:
        write(run / name, {})
    manifests = {name: {"format": "test-fixture", "files_sha256": {"model.safetensors": "hash-only"}} for name in ("initial", "final")}
    for name, manifest in manifests.items():
        folder = run / (name + "_reader")
        folder.mkdir()
        write(folder / "reader_manifest.json", manifest)
    policy = {"format": "cpu-reader-policy-v1", "selection_uses_test": False,
              "selection_sha256": exporter.digest(run / "selection.json"),
              "protocol_sha256": exporter.digest(run / "protocol.json"),
              "reader_manifest_sha256": {name: exporter.digest(run / (name + "_reader") / "reader_manifest.json") for name in manifests}}
    write(run / "policy.json", policy)
    report = {"format": "cpu-reader-report-v1", "reader_manifests": manifests,
              "artifact_sha256": {name: exporter.digest(run / name) for name in exporter.ALLOWED}}
    write(run / "report.json", report)
    return run


def test_export_copies_only_explicit_bound_files_and_is_idempotent(tmp_path):
    run = completed_fixture(tmp_path)
    (run / "raw-local.json").write_text('{"prediction":"private"}', encoding="utf-8")
    (run / "model.safetensors").write_bytes(b"not-for-publication")
    output = tmp_path / "public"
    first = exporter.export(run, output)
    assert set(first) == exporter.ALLOWED | {"report.json"}
    assert exporter.export(run, output) == first
    assert not (output / "raw-local.json").exists()
    assert not (output / "model.safetensors").exists()


def test_export_rejects_conflicting_existing_bytes_before_copying(tmp_path):
    run = completed_fixture(tmp_path)
    output = tmp_path / "public"
    output.mkdir()
    (output / "policy.json").write_text("preserve", encoding="utf-8")
    with pytest.raises(FileExistsError, match="preserved"):
        exporter.export(run, output)
    assert {path.name for path in output.iterdir()} == {"policy.json"}
    assert (output / "policy.json").read_text() == "preserve"


@pytest.mark.parametrize("tamper", ["checksum", "allowlist", "manifest", "benchmark", "raw_field"])
def test_export_rejects_unbound_or_private_evidence(tmp_path, tamper):
    run = completed_fixture(tmp_path)
    if tamper == "checksum":
        write(run / "training_curve.json", {"changed": True})
    elif tamper == "allowlist":
        report = exporter.read(run / "report.json")
        report["artifact_sha256"]["model.safetensors"] = "extra"
        write(run / "report.json", report)
    elif tamper == "manifest":
        write(run / "initial_reader/reader_manifest.json", {"changed": True})
    elif tamper == "benchmark":
        write(run / "benchmark-reader.json", {"format": "cpu-reader-benchmark-v1", "mode": "reader",
                                              "source_report_sha256": "wrong", "policy_sha256": "wrong"})
    else:
        write(run / "training_curve.json", {"span_prediction": "raw source text"})
        report = exporter.read(run / "report.json")
        report["artifact_sha256"]["training_curve.json"] = exporter.digest(run / "training_curve.json")
        write(run / "report.json", report)
    with pytest.raises(ValueError):
        exporter.export(run, tmp_path / "public")
    assert not (tmp_path / "public").exists()


def test_independent_retention_counts_intersection_not_equal_totals():
    original = [{"id": "a", "is_impossible": False, "exact_match": 1},
                {"id": "b", "is_impossible": False, "exact_match": 0},
                {"id": "missing", "is_impossible": True, "exact_match": 0}]
    candidate = [{"id": "a", "is_impossible": False, "exact_match": 0},
                 {"id": "b", "is_impossible": False, "exact_match": 1},
                 {"id": "missing", "is_impossible": True, "exact_match": 1}]
    retained = audit.paired_retention(original, candidate)
    assert retained["baseline_exact_correct"] == retained["candidate_exact_correct"] == 1
    assert retained["lost_exact_correct"] == retained["new_exact_correct"] == 1
    assert retained["exact_correct_retention"] == 0


def test_safetensor_state_hash_is_reconstructed_without_model_loading(tmp_path):
    path = tmp_path / "tiny.safetensors"
    first, second = struct.pack("<ff", 1.0, 2.0), struct.pack("<f", 3.0)
    header = {"z": {"dtype": "F32", "shape": [1], "data_offsets": [8, 12]},
              "a": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    encoded = json.dumps(header).encode()
    path.write_bytes(len(encoded).to_bytes(8, "little") + encoded + first + second)
    expected = hashlib.sha256(b"a" + first + b"z" + second).hexdigest()
    assert audit.tensor_state(path) == (expected, 3)


def test_privacy_checks_reject_private_paths_but_accept_source_urls():
    evidence = audit.Audit()
    audit.privacy(evidence, {"source_url": "https://example.org/source.json", "files_sha256": {"model.safetensors": "hash"}})
    with pytest.raises(ValueError, match="private absolute"):
        audit.privacy(evidence, {"path": "C:\\Users\\private\\weights.bin"})


def test_checkpoint_ties_keep_initial_and_rejection_keeps_generator():
    initial = {"accepted": True, "selected_threshold": 0.5,
               "selected": {"answerable_token_f1": 0.8, "correct_abstention_recall": 0.5}}
    assert audit.choose_checkpoint({"initial": initial, "final": initial})["checkpoint"] == "initial"
    rejected = {"accepted": False, "diagnostic_threshold": "always_answer",
                "diagnostic_candidate": {"gate": {"retention_and_f1_pass": False, "exact_correct_retention": 0.8},
                                         "metrics": {"answerable_token_f1": 0.7, "correct_abstention_recall": 0.0}}}
    result = audit.choose_checkpoint({"initial": rejected, "final": rejected})
    assert result["kind"] == "generator" and result["checkpoint"] is None and result["diagnostic_checkpoint"] == "initial"


@pytest.mark.parametrize("mutation", [None, "sizes", "count", "width", "padded_tokens"])
def test_microbatch_audit_checks_global_padding_and_actual_partition(mutation):
    by_id = {"a": {"reader_tokens": 5}, "b": {"reader_tokens": 7}, "c": {"reader_tokens": 8}, "d": {"reader_tokens": 9}}
    item = {"microbatch_sizes": [2, 2], "microbatch_count": 2, "global_padded_width": 9, "padded_tokens": 36}
    if mutation == "sizes":
        item["microbatch_sizes"] = [1, 3]
    elif mutation == "count":
        item["microbatch_count"] = 1
    elif mutation == "width":
        item["global_padded_width"] = 8
    elif mutation == "padded_tokens":
        item["padded_tokens"] = 32
    config = {"training_batch_size": 4, "training_microbatch_size": 2}
    if mutation is None:
        audit.audit_microbatch(audit.Audit(), item, list(by_id), by_id, config)
    else:
        with pytest.raises(ValueError, match="Audit failed"):
            audit.audit_microbatch(audit.Audit(), item, list(by_id), by_id, config)


def test_execution_comparison_is_optional_for_independent_reproduction():
    cfg = {"training_microbatch_size": 4}
    args = SimpleNamespace(calibrations=None, prior_run=None)
    assert audit.audit_execution_comparison(audit.Audit(), args, cfg, None, None, None, None, None) is None
    args.calibrations = Path("calibrations")
    with pytest.raises(ValueError, match="requires both"):
        audit.audit_execution_comparison(audit.Audit(), args, cfg, None, None, None, None, None)
