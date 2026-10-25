"""Fresh SQuAD2 passages for an answer verifier, excluding prior experiments.

The first historical manifest defines the original adapter's train/validation
article holdout. Every supplied manifest contributes IDs and contexts excluded
from every new split. Only original passages and native answerability are used.
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

from . import document_v2_data as base
from .training import write_json

DATA_KEYS = base.DATA_KEYS
STRATA = base.STRATA
MARKER = base.MARKER
prompt_text = base.prompt_text
stratum = base.stratum
public_manifest = base.public_manifest
validate_splits = base.validate_splits
sha256 = base.sha256


def exclusion_controls(manifests: list[dict], manifest_hashes: list[str]) -> dict:
    if not manifests or len(manifests) != len(manifest_hashes):
        raise ValueError("At least one ordered prior manifest and its checksum are required.")
    if len(set(manifest_hashes)) != len(manifest_hashes):
        raise ValueError("Duplicate prior manifests are not allowed.")
    for manifest in manifests:
        base.prior_controls(manifest)  # Validates the three nonempty split lists.
    records = [row for manifest in manifests for split in ("train", "validation", "test") for row in manifest[split]]
    return {
        "prior_manifests": [{"index": index, "sha256": digest,
                             "split_counts": {name: len(manifest[name]) for name in ("train", "validation", "test")}}
                            for index, (manifest, digest) in enumerate(zip(manifests, manifest_hashes, strict=True))],
        "original_article_holdout_manifest_index": 0,
        "original_article_holdout_manifest_sha256": manifest_hashes[0],
        "ancestor_train_and_validation_article_sha256": sorted({row["article_sha256"] for split in ("train", "validation") for row in manifests[0][split]}),
        "all_prior_ids": sorted({row["id"] for row in records}),
        "all_prior_context_sha256": sorted({row["context_sha256"] for row in records}),
        "all_prior_article_sha256": sorted({row["article_sha256"] for row in records}),
        "prior_record_count_with_repetition": len(records),
    }


def fresh_rows(rows: list[dict], controls: dict) -> tuple[list[dict], dict]:
    identifiers = set(controls["all_prior_ids"])
    contexts = set(controls["all_prior_context_sha256"])
    selected = [row for row in rows if row["id"] not in identifiers and row["context_sha256"] not in contexts]
    statistics = {
        "eligible_questions_before_historical_exclusions": len(rows),
        "questions_matching_prior_ids": sum(row["id"] in identifiers for row in rows),
        "questions_matching_prior_contexts": sum(row["context_sha256"] in contexts for row in rows),
        "questions_excluded_by_either_rule": len(rows) - len(selected),
        "fresh_questions": len(selected),
        "fresh_distinct_contexts": len({row["context_sha256"] for row in selected}),
        "fresh_distinct_contexts_by_stratum": {group: len({row["context_sha256"] for row in selected if stratum(row) == group}) for group in STRATA},
    }
    return selected, statistics


def check_historical_exclusions(splits: dict, controls: dict) -> None:
    identifiers = set(controls["all_prior_ids"])
    contexts = set(controls["all_prior_context_sha256"])
    ancestor_articles = set(controls["ancestor_train_and_validation_article_sha256"])
    for name, rows in splits.items():
        if any(row["id"] in identifiers or row["context_sha256"] in contexts for row in rows):
            raise ValueError(f"Verifier {name} repeats a prior question ID or context.")
    if any(row["article_sha256"] in ancestor_articles for row in splits["validation"] + splits["test"]):
        raise ValueError("Verifier held-out articles overlap the original adapter's train/validation pool.")


def select_splits(train_rows: list[dict], dev_rows: list[dict], config: dict, controls: dict) -> tuple[dict, dict]:
    fresh_train, train_exclusions = fresh_rows(train_rows, controls)
    fresh_dev, dev_exclusions = fresh_rows(dev_rows, controls)
    original_controls = {
        "prior_train_and_validation_article_sha256": controls["ancestor_train_and_validation_article_sha256"],
        "prior_test_ids": [], "prior_test_context_sha256": [],
        "prior_test_article_sha256": controls["all_prior_article_sha256"],
    }
    # Reuse the tested article split, scarce-stratum-first balanced sampler and
    # one-question-per-context checks after applying the broader fresh-data rule.
    splits, selection = base.select_splits(fresh_train, fresh_dev, config, original_controls)
    check_historical_exclusions(splits, controls)
    selection.pop("prior_test_rows_excluded_by_id_or_context")
    selection.pop("prior_test_contexts_excluded")
    selection.pop("selected_test_articles_previously_evaluated")
    previous_articles = set(controls["all_prior_article_sha256"])
    selection["historical_exclusion_counts"] = {"official_train": train_exclusions, "official_dev": dev_exclusions}
    selection["historical_article_reuse"] = {
        name: {"selected_articles_seen_in_any_prior_manifest": len({row["article_sha256"] for row in rows} & previous_articles),
               "examples_on_previously_used_articles": sum(row["article_sha256"] in previous_articles for row in rows)}
        for name, rows in splits.items()
    }
    ancestor_articles = set(controls["ancestor_train_and_validation_article_sha256"])
    fresh_pools = {"train": [row for row in fresh_train if row["article_sha256"] in ancestor_articles],
                   "validation": [row for row in fresh_train if row["article_sha256"] not in ancestor_articles],
                   "test": fresh_dev}
    availability = {}
    for name, rows in fresh_pools.items():
        by_stratum = {group: len({row["context_sha256"] for row in rows if stratum(row) == group}) for group in STRATA}
        by_length = {group: len({row["context_sha256"] for row in rows if row["length_group"] == group}) for group in ("short", "long")}
        availability[name] = {"fresh_distinct_contexts_by_stratum": by_stratum,
                              "fresh_distinct_contexts_by_length": by_length,
                              "balanced_size_upper_bound": min(4 * min(by_stratum.values()), 4 * (by_length["short"] // 2), 4 * (by_length["long"] // 2)),
                              "selected_size_proven_feasible": len(splits[name])}
    selection["availability_before_generation"] = availability
    return splits, selection


def prepare(config: dict, output: Path, cache: Path, download: bool, prior_manifests: list[Path]) -> tuple[dict, dict]:
    from transformers import AutoTokenizer
    base.validate_config(config)
    if any((output / name).exists() for name in ("data", "split_manifest.json", "data_provenance.json")):
        raise FileExistsError("Previous verifier data is preserved; choose a new output directory.")
    payloads = [Path(path).read_bytes() for path in prior_manifests]
    controls = exclusion_controls([json.loads(payload) for payload in payloads], [sha256(payload) for payload in payloads])
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["revision"], local_files_only=True, trust_remote_code=False)
    sources = base.read_sources(cache, download)
    train, train_stats = base.eligible_records(sources["train-v2.0.json"], tokenizer, config, "official_train")
    dev, dev_stats = base.eligible_records(sources["dev-v2.0.json"], tokenizer, config, "official_dev_reserved_as_local_test")
    splits, selection = select_splits(train, dev, config, controls)
    data_directory = output / "data"
    data_directory.mkdir(parents=True)
    local_hashes = {}
    for name, rows in splits.items():
        path = data_directory / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")
        local_hashes[name] = sha256(path.read_bytes())
    write_json(output / "split_manifest.json", public_manifest(splits))
    provenance = {
        "dataset": "Fresh SQuAD2.0 verifier subset with all historical experiment questions and contexts excluded",
        "source_revision": base.SOURCE_REVISION,
        "source_urls": {name: base.SOURCE_BASE + "/" + name for name in base.SOURCES},
        "source_sha256": base.SOURCES,
        "citation": "Rajpurkar, Jia, Liang (2018). Know What You Don't Know: Unanswerable Questions for SQuAD. arXiv:1806.03822.",
        "license": "CC BY-SA 4.0", "license_source": "https://rajpurkar.github.io/SQuAD-explorer/",
        "data_config": {key: config[key] for key in DATA_KEYS},
        "prior_manifest_sha256": [entry["sha256"] for entry in controls["prior_manifests"]],
        "eligibility": {"official_train": train_stats, "official_dev": dev_stats},
        "historical_exclusion_controls": controls,
        "selection": selection, "counts": {name: len(rows) for name, rows in splits.items()},
        "local_data_sha256": local_hashes,
        "split_manifest_sha256": sha256((output / "split_manifest.json").read_bytes()),
        "policy": "All supplied historical manifests contribute every train/validation/test question ID and context hash to exclusions for all new splits. The first manifest alone defines the original adapter's train/validation article pool: unseen official-training articles remain verifier validation; remaining official-training articles form verifier training; official-development articles form verifier test. Current splits are article-disjoint and contain one question per distinct context, balanced across native answerability and length. No answer-aware cropping, answer removal or synthetic negatives.",
        "availability_selection": "Requested counts are fixed before model generation. Validation/test reductions, when needed, use only fresh distinct-context availability and the declared deterministic sampler, never model outcomes.",
        "initial_experiment_targets": {"train": 1024, "validation": 256, "test": 512},
        "count_reduction_rationale": ("The initial experiment requested 256 validation and 512 test examples. "
            f"After historical exclusions this preparation has {selection['availability_before_generation']['validation']['fresh_distinct_contexts_by_stratum']['missing_long']} fresh missing-long validation contexts "
            f"and {selection['availability_before_generation']['test']['fresh_distinct_contexts_by_length']['long']} distinct long test contexts. "
            f"Balanced upper bounds are {selection['availability_before_generation']['validation']['balanced_size_upper_bound']} for validation "
            f"and {selection['availability_before_generation']['test']['balanced_size_upper_bound']} for test. "
            "The deterministic sampler confirms the configured actual sizes are feasible. Counts were chosen from data availability before any model generation."),
        "limitations": ["Question IDs and exact contexts are fresh relative to the supplied manifests, but article topics were used in prior experiments; historical topic reuse is recorded.",
                        "Validation reuses the original held-out article pool with fresh passages. This is not a corpus-level independent replication.",
                        "Balanced filtered sample, not native prevalence, the complete benchmark or its hidden official test.",
                        "Exact exclusions do not remove all paraphrases, related facts, near duplicates or possible base-model pretraining exposure.",
                        f"Longer passages mean {config['short_context_tokens'] + 1}-{config['max_context_tokens']} model tokens, not multi-page documents."],
    }
    write_json(output / "data_provenance.json", provenance)
    return splits, provenance


def load_prepared(output: Path, config: dict) -> tuple[dict, dict]:
    base.validate_config(config)
    provenance = json.loads((output / "data_provenance.json").read_bytes())
    controls = provenance["historical_exclusion_controls"]
    prior_hashes = [entry["sha256"] for entry in controls["prior_manifests"]]
    if not prior_hashes or provenance["prior_manifest_sha256"] != prior_hashes or controls["original_article_holdout_manifest_sha256"] != prior_hashes[0]:
        raise ValueError("Prepared verifier prior manifest bindings disagree.")
    if provenance["data_config"] != {key: config[key] for key in DATA_KEYS}:
        raise ValueError("Prepared verifier data configuration differs from this run.")
    manifest_bytes = (output / "split_manifest.json").read_bytes()
    if sha256(manifest_bytes) != provenance["split_manifest_sha256"]:
        raise ValueError("Prepared verifier manifest checksum mismatch.")
    splits = {}
    for name in ("train", "validation", "test"):
        payload = (output / "data" / f"{name}.jsonl").read_bytes()
        if sha256(payload) != provenance["local_data_sha256"][name]:
            raise ValueError("Prepared verifier local data checksum mismatch.")
        splits[name] = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
        if len(splits[name]) != config[f"{name}_size"]:
            raise ValueError("Prepared verifier split count differs from configuration.")
    validate_splits(splits)
    if public_manifest(splits) != json.loads(manifest_bytes):
        raise ValueError("Prepared verifier rows differ from their public manifest.")
    check_historical_exclusions(splits, controls)
    for rows in splits.values():
        for row in rows:
            group = "short" if row["context_tokens"] <= config["short_context_tokens"] else "long"
            if row["length_group"] != group or not 0 < row["context_tokens"] <= config["max_context_tokens"] or row["prompt_tokens"] > config["max_prompt_tokens"]:
                raise ValueError("Prepared verifier length metadata exceeds configured limits.")
    return splits, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/document-verifier.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache/document-v2-data"))
    parser.add_argument("--prior-manifests", type=Path, nargs="+", default=[Path("results/document-split-manifest.json"), Path("results/document-v2-split-manifest.json")])
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Accepted for symmetry; this command only prepares data.")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose a new verifier data output directory.")
    config = json.loads(args.config.read_bytes())
    _, provenance = prepare(config, args.output, args.cache, args.download_data, args.prior_manifests)
    write_json(args.output / "resolved_config.json", config)
    write_json(args.output / "run_status.json", {"status": "prepared"})
    print(json.dumps({"counts": provenance["counts"], "selection": provenance["selection"]}, indent=2))


if __name__ == "__main__":
    main()
