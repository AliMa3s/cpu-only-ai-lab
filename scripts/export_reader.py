"""Export only checksum-bound numerical evidence from a completed reader run."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ALLOWED = {
    "resolved_config.json", "protocol.json", "selection.json", "policy.json",
    "data_provenance.json", "split_manifest.json", "sample_schedule.json", "training_curve.json",
    "generator-validation.json", "generator-test.json", "initial-validation-decoded.json",
    "final-validation-decoded.json", "initial-test-decoded.json", "final-test-decoded.json",
}
BENCHMARKS = {"benchmark-paired.json": "paired", "benchmark-generator.json": "generator", "benchmark-reader.json": "reader"}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def public_fields_only(value):
    forbidden = {"context", "question", "prediction", "span_prediction", "references", "response", "movie_id", "article", "answer", "input_ids", "raw_logits"}
    if isinstance(value, dict):
        if set(value) & forbidden:
            raise ValueError("Public evidence contains a raw source/generation field.")
        for child in value.values():
            public_fields_only(child)
    elif isinstance(value, list):
        for child in value:
            public_fields_only(child)


def export(run: Path, output: Path):
    if read(run / "run_status.json").get("status") != "complete":
        raise ValueError("Only a completed reader run can be published.")
    report, policy = read(run / "report.json"), read(run / "policy.json")
    if report.get("format") != "cpu-reader-report-v1" or set(report["artifact_sha256"]) != ALLOWED:
        raise ValueError("Unexpected reader publication artifact list.")
    if policy.get("format") != "cpu-reader-policy-v1" or policy.get("selection_uses_test") is not False:
        raise ValueError("Invalid validation-only reader policy.")
    names = ["report.json"] + sorted(ALLOWED)
    for name, expected in report["artifact_sha256"].items():
        if digest(run / name) != expected:
            raise ValueError("Reader publication artifact checksum mismatch: " + name)
    for name in ("selection", "protocol"):
        if digest(run / (name + ".json")) != policy[name + "_sha256"]:
            raise ValueError("Reader policy source binding mismatch.")
    for checkpoint in ("initial", "final"):
        path = run / (checkpoint + "_reader") / "reader_manifest.json"
        if digest(path) != policy["reader_manifest_sha256"][checkpoint] or read(path) != report["reader_manifests"][checkpoint]:
            raise ValueError("Reader checkpoint manifest differs from the completed policy/report.")
    for name, mode in BENCHMARKS.items():
        path = run / name
        if not path.exists():
            continue
        benchmark = read(path)
        if (benchmark.get("format") != "cpu-reader-benchmark-v1" or benchmark.get("mode") != mode or
                benchmark["source_report_sha256"] != digest(run / "report.json") or
                benchmark["policy_sha256"] != digest(run / "policy.json")):
            raise ValueError("Reader benchmark source binding differs.")
        names.append(name)
    # Validate every source and every pre-existing destination before writing.
    for name in names:
        public_fields_only(read(run / name))
        if (output / name).exists() and (output / name).read_bytes() != (run / name).read_bytes():
            raise FileExistsError("Existing public reader evidence is preserved.")
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        (output / name).write_bytes((run / name).read_bytes())
    return {name: digest(output / name) for name in names}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("artifacts/reader-run"))
    parser.add_argument("--output", type=Path, default=Path("results/reader"))
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.output), indent=2))
