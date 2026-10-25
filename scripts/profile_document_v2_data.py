"""Audit public document split manifests without importing model libraries."""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics


def read(path: Path):
    return json.loads(path.read_bytes())


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def overlap(current, previous):
    result = {}
    for key in ("id", "context_sha256", "article_sha256"):
        known = {row[key] for row in previous}
        current_values = {row[key] for row in current}
        result[key] = {"current_rows_matching": sum(row[key] in known for row in current),
                       "unique_shared": len(known & current_values)}
    return result


def reconstruct_selected(prior_rows, config, metrics):
    """Replay document_task.py's seeded shuffled stream through selected step."""
    selected_step = metrics["selection"]["step"]
    if not 0 <= selected_step <= config["optimizer_steps"] or len(prior_rows) != config["train_size"]:
        raise ValueError("Prior selected-step or dataset count is inconsistent.")
    rng = random.Random(config["seed"])
    order = list(range(len(prior_rows)))
    rng.shuffle(order)
    position, ordered_ids, seen, reshuffles = 0, [], set(), 0
    for _ in range(selected_step * config["batch_size"]):
        if position == len(order):
            rng.shuffle(order)
            position = 0
            reshuffles += 1
        index = order[position]
        position += 1
        seen.add(index)
        ordered_ids.append(prior_rows[index]["id"])
    rows = [prior_rows[index] for index in sorted(seen)]
    notes = {"selected_step": selected_step, "batch_size": config["batch_size"], "seed": config["seed"],
             "presentations_before_selected_checkpoint": len(ordered_ids), "unique_examples_before_selected_checkpoint": len(rows),
             "ordered_selected_ids_sha256": hashlib.sha256(json.dumps(ordered_ids, separators=(",", ":")).encode()).hexdigest(),
             "method": "Reconstruct document_task.py sampling: shuffle indices with random.Random(seed), consume sequentially, reshuffle the same index list when exhausted; stop after selected_step * batch_size presentations.",
             "reshuffles_after_initial_order": reshuffles,
             "evidence_limit": f"Reconstructed from saved configuration, selected step and original manifest order; the prior run did not save an observed per-update sample-ID trace. For this run, step {selected_step} * batch {config['batch_size']} = {len(ordered_ids)} presentations from {len(prior_rows)} prior training rows, with {reshuffles} reshuffles after the initial order."}
    return rows, notes


def build(args):
    paths = {"current_manifest": args.current_manifest, "current_provenance": args.current_provenance,
             "prior_manifest": args.prior_manifest, "prior_config": args.prior_config,
             "prior_metrics": args.prior_metrics, "prior_training_source": args.prior_training_source}
    current, provenance, prior, config, metrics = [read(paths[key]) for key in
        ("current_manifest", "current_provenance", "prior_manifest", "prior_config", "prior_metrics")]
    hashes = {name: digest(path) for name, path in paths.items()}
    if hashes["current_manifest"] != provenance["split_manifest_sha256"]:
        raise ValueError("Current manifest does not match its data provenance.")
    if hashes["prior_manifest"] != provenance["prior_manifest_sha256"] or hashes["prior_manifest"] != metrics["dataset"]["split_manifest_sha256"]:
        raise ValueError("Prior manifest does not match both experiment records.")
    if hashes["prior_config"] != metrics["resolved_config_sha256"]:
        raise ValueError("Prior configuration hash differs from saved metrics.")
    selected, reconstruction = reconstruct_selected(prior["train"], config, metrics)
    comparisons = {"all_prior_available_training_rows": prior["train"],
                   "prior_selected_checkpoint_training_rows_reconstructed": selected,
                   "prior_validation": prior["validation"], "prior_test": prior["test"]}
    intersections = {name: overlap(current["train"], values) for name, values in comparisons.items()}
    inherited_articles = {row["article_sha256"] for split in ("train", "validation") for row in prior[split]}
    old_test_ids = {row["id"] for row in prior["test"]}
    old_test_contexts = {row["context_sha256"] for row in prior["test"]}
    checks = {
        "current_validation_excludes_all_prior_training_and_validation_articles": all(row["article_sha256"] not in inherited_articles for row in current["validation"]),
        "current_test_excludes_all_prior_training_and_validation_articles": all(row["article_sha256"] not in inherited_articles for row in current["test"]),
        "current_test_excludes_prior_test_ids": all(row["id"] not in old_test_ids for row in current["test"]),
        "current_test_excludes_prior_test_contexts": all(row["context_sha256"] not in old_test_contexts for row in current["test"]),
    }
    lengths, counts = {}, {}
    observed = {key: set() for key in ("article_sha256", "context_sha256", "id")}
    for name in ("train", "validation", "test"):
        rows = current[name]
        counts[name] = {"examples": len(rows), "articles": len({row["article_sha256"] for row in rows}),
                        "strata": dict(Counter(("missing_" if row["is_impossible"] else "answerable_") + row["length_group"] for row in rows))}
        checks[name + "_four_equal_strata"] = len(counts[name]["strata"]) == 4 and len(set(counts[name]["strata"].values())) == 1
        checks[name + "_unique_ids_and_contexts"] = all(len({row[key] for row in rows}) == len(rows) for key in ("id", "context_sha256"))
        for key in observed:
            values = {row[key] for row in rows}
            checks[name + "_disjoint_" + key] = not bool(observed[key] & values)
            observed[key].update(values)
        lengths[name] = {}
        for group in ("short", "long"):
            values = [row["context_tokens"] for row in rows if row["length_group"] == group]
            lengths[name][group] = {"examples": len(values), "minimum": min(values), "median": statistics.median(values),
                                    "maximum": max(values), "total_context_tokens": sum(values)}
    if not all(checks.values()):
        raise ValueError("Prepared split audit failed: " + ", ".join(name for name, value in checks.items() if not value))
    return {"format": "cpu-ai-lab-document-v2-data-audit-v1", "source_sha256": hashes,
            "method": "Read-only analysis of public-schema manifests, provenance and prior run metadata; no raw text, tokenizer, model or training execution.",
            "counts": counts, "context_token_lengths": lengths,
            "prior_selected_training_reconstruction": reconstruction,
            "current_training_overlap": intersections, "checks": checks, "all_checks_passed": True,
            "new_test_previous_article_overlap": overlap(current["test"], prior["test"])["article_sha256"],
            "limitations": ["Continued adaptation intentionally reuses training topics and some prior examples; the current row count does not imply entirely new contexts.",
                            "New test questions and contexts differ from the previous test, but source articles were evaluated previously; the overlap count is recorded above.",
                            "Exact hash/ID separation does not establish factual independence or rule out semantic near duplicates and base-model pretraining exposure.",
                            f"Longer passages here remain at most {provenance['data_config']['max_context_tokens']} model tokens under the recorded configuration, not multi-page documents."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-manifest", type=Path, default=Path("artifacts/document-v2-data/split_manifest.json"))
    parser.add_argument("--current-provenance", type=Path, default=Path("artifacts/document-v2-data/data_provenance.json"))
    parser.add_argument("--prior-manifest", type=Path, default=Path("results/document-split-manifest.json"))
    parser.add_argument("--prior-config", type=Path, default=Path("results/document-config.json"))
    parser.add_argument("--prior-metrics", type=Path, default=Path("results/document-metrics.json"))
    parser.add_argument("--prior-training-source", type=Path, default=Path("src/cpu_ai_lab/document_task.py"))
    parser.add_argument("--output", type=Path, default=Path("results/document-v2-data-audit.json"))
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Existing audit reports are preserved; select a new output path.")
    result = build(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(json.dumps({"all_checks_passed": result["all_checks_passed"],
                      "prior_selected_training": result["prior_selected_training_reconstruction"],
                      "current_training_overlap": result["current_training_overlap"]}, indent=2))


if __name__ == "__main__":
    main()
