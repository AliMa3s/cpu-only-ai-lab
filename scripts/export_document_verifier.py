"""Publish only checksum-bound numerical artifacts from a completed verifier run."""
import argparse
import hashlib
import json
from pathlib import Path


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def export(run, output):
    if json.loads((run / "run_status.json").read_bytes())["status"] != "complete":
        raise ValueError("A completed run is required.")
    report = json.loads((run / "report.json").read_bytes())
    names = ["report.json"]
    allowed = {"resolved_config.json", "protocol.json", "selection.json", "policy.json", "verifier.json",
               "data_provenance.json", "split_manifest.json", "train-features.json", "validation-features.json", "test-features.json"}
    if set(report["artifact_sha256"]) != allowed:
        raise ValueError("Unexpected publication artifact list.")
    for name, expected in report["artifact_sha256"].items():
        if digest(run / name) != expected:
            raise ValueError(f"Artifact checksum mismatch: {name}")
        names.append(name)
    for mode in ("paired", "baseline", "verifier"):
        name = f"benchmark-{mode}.json"
        if (run / name).exists():
            evidence = json.loads((run / name).read_bytes())
            if evidence["source_report_sha256"] != digest(run / "report.json") or evidence["policy_sha256"] != digest(run / "policy.json"):
                raise ValueError("Benchmark source binding differs.")
            names.append(name)
    for name in names:
        if (output / name).exists() and (output / name).read_bytes() != (run / name).read_bytes():
            raise FileExistsError("Existing publication bytes are preserved.")
    output.mkdir(parents=True, exist_ok=True)
    for name in names:
        (output / name).write_bytes((run / name).read_bytes())
    return {name: digest(output / name) for name in names}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("artifacts/document-verifier-run"))
    parser.add_argument("--output", type=Path, default=Path("results/document-verifier"))
    args = parser.parse_args()
    print(json.dumps(export(args.run, args.output), indent=2))
