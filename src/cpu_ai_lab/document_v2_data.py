"""Pinned SQuAD2 data for longer passages and native no-answer questions.

Raw text stays in ignored local artifacts. Public manifests contain only source
IDs, labels, token counts and hashes. No passage is cropped around a gold answer.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import urllib.request

from .document_task import normalize_answer, sha256
from .training import render_prompt, write_json

SOURCE_REVISION = "e0c66cfec263165fb3c15cb76a22175e371b71d7"
SOURCE_BASE = f"https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/{SOURCE_REVISION}/dataset"
SOURCES = {
    "train-v2.0.json": "68dcfbb971bd3e96d5b46c7177b16c1a4e7d4bdef19fb204502738552dede002",
    "dev-v2.0.json": "80a5225e94905956a6446d296ca1093975c4d3b3260f1d6c8f68bc2ab77182d8",
}
STRATA = ("answerable_short", "answerable_long", "missing_short", "missing_long")
MARKER = "NOT_FOUND"
DATA_KEYS = ("model_id", "revision", "seed", "train_size", "validation_size", "test_size",
             "max_context_tokens", "short_context_tokens", "max_prompt_tokens", "max_answer_tokens",
             "max_sequence_length")
INSTRUCTION = ("Answer the question using the passage. Copy only the shortest answer span. "
               "If the passage does not contain the answer, output exactly NOT_FOUND. "
               "Do not add explanations.\nPassage:\n")


def prompt_text(context: str, question: str) -> str:
    return INSTRUCTION + context + "\nQuestion: " + question


def stratum(row: dict) -> str:
    if type(row["is_impossible"]) is not bool or row["length_group"] not in ("short", "long"):
        raise ValueError("A row requires native answerability and short/long length group.")
    return ("missing_" if row["is_impossible"] else "answerable_") + row["length_group"]


def validate_config(config: dict) -> None:
    for key in DATA_KEYS:
        if key not in config:
            raise ValueError(f"Missing dataset setting: {key}")
    for key in DATA_KEYS[2:]:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Dataset setting {key} must be a positive integer.")
    if not config["short_context_tokens"] < config["max_context_tokens"]:
        raise ValueError("The long-passage group must have a nonempty token interval.")
    if config["max_prompt_tokens"] + config["max_answer_tokens"] + 1 > config["max_sequence_length"]:
        raise ValueError("Prompt and answer budgets do not fit the sequence budget.")
    for name in ("train", "validation", "test"):
        if config[f"{name}_size"] % len(STRATA):
            raise ValueError("Each split size must divide equally into four strata.")


def read_sources(cache: Path, download: bool) -> dict:
    documents = {}
    for name, expected in SOURCES.items():
        path = cache / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {name}; pass --download-data once.")
            cache.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(SOURCE_BASE + "/" + name, timeout=60) as response:
                payload = response.read(60_000_001)
            if len(payload) > 60_000_000 or sha256(payload) != expected:
                raise ValueError("Downloaded SQuAD2 file failed its pinned checksum/size check.")
            path.write_bytes(payload)
        payload = path.read_bytes()
        if sha256(payload) != expected:
            raise ValueError(f"Pinned SQuAD2 checksum mismatch: {name}")
        value = json.loads(payload)
        if value.get("version") != "v2.0":
            raise ValueError("The native SQuAD2.0 source version is required.")
        documents[name] = value["data"]
    return documents


def prior_controls(manifest: dict) -> dict:
    for split in ("train", "validation", "test"):
        if not isinstance(manifest.get(split), list) or not manifest[split]:
            raise ValueError("The previous document split manifest is incomplete.")
    return {
        "prior_train_and_validation_article_sha256": sorted({r["article_sha256"] for s in ("train", "validation") for r in manifest[s]}),
        "prior_test_ids": sorted({r["id"] for r in manifest["test"]}),
        "prior_test_context_sha256": sorted({r["context_sha256"] for r in manifest["test"]}),
        "prior_test_article_sha256": sorted({r["article_sha256"] for r in manifest["test"]}),
    }


def eligible_records(documents: list[dict], tokenizer, config: dict, source: str) -> tuple[list[dict], dict]:
    rows, stats = [], Counter()
    token_cache = {}
    marker_tokens = len(tokenizer.encode(MARKER, add_special_tokens=False))
    if marker_tokens > config["max_answer_tokens"]:
        raise ValueError("The NOT_FOUND marker does not fit the declared answer budget.")
    for article in documents:
        title = article["title"]
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            context_hash = sha256(context.encode("utf-8"))
            stats["source_paragraphs"] += 1
            if context_hash not in token_cache:
                token_cache[context_hash] = len(tokenizer.encode(context, add_special_tokens=False))
            context_tokens = token_cache[context_hash]
            for question in paragraph["qas"]:
                stats["source_questions"] += 1
                impossible = question.get("is_impossible")
                if type(impossible) is not bool:
                    raise ValueError("Every source question must carry native boolean is_impossible.")
                if impossible == bool(question["answers"]):
                    raise ValueError("Native answerability disagrees with the official answer list.")
                if not 0 < context_tokens <= config["max_context_tokens"]:
                    stats["rejected_context_length"] += 1
                    continue
                references = []
                for answer in question["answers"]:
                    text, start = answer["text"], answer["answer_start"]
                    if context[start:start + len(text)] != text:
                        raise ValueError("Gold answer is not at its declared source span.")
                    if text not in references:
                        references.append(text)
                if any(answer.strip() == MARKER for answer in references):
                    stats["rejected_ambiguous_marker_reference"] += 1
                    continue
                reference_lengths = [len(tokenizer.encode(text, add_special_tokens=False)) for text in references]
                if any(length > config["max_answer_tokens"] or not normalize_answer(text)
                       for text, length in zip(references, reference_lengths)):
                    stats["rejected_answer_length_or_empty_normalization"] += 1
                    continue
                prompt = prompt_text(context, question["question"])
                prompt_tokens = len(tokenizer.encode(render_prompt(tokenizer, {"prompt": prompt}), add_special_tokens=False))
                if prompt_tokens > config["max_prompt_tokens"]:
                    stats["rejected_prompt_length"] += 1
                    continue
                target_length = marker_tokens if impossible else reference_lengths[0]
                if prompt_tokens + target_length + 1 > config["max_sequence_length"]:
                    raise ValueError("Declared target and prompt do not fit the training sequence budget.")
                row = {"id": question["id"], "article": title, "source": source,
                       "context": context, "question": question["question"], "prompt": prompt,
                       "response": MARKER if impossible else references[0], "references": references,
                       "is_impossible": impossible,
                       "length_group": "short" if context_tokens <= config["short_context_tokens"] else "long",
                       "article_sha256": sha256(title.encode("utf-8")), "context_sha256": context_hash,
                       "question_sha256": sha256(question["question"].encode("utf-8")),
                       "prompt_tokens": prompt_tokens, "context_tokens": context_tokens,
                       "reference_token_counts": reference_lengths}
                rows.append(row)
                stats["eligible_questions"] += 1
                stats["eligible_" + stratum(row)] += 1
    stats["eligible_articles"] = len({row["article_sha256"] for row in rows})
    stats["eligible_contexts"] = len({row["context_sha256"] for row in rows})
    return rows, dict(stats)


def sample_stratified(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Balance all four strata while retaining only one Q per context globally."""
    if type(count) is not int or count < 4 or count % 4:
        raise ValueError("A balanced sample requires a positive multiple of four.")
    rng = random.Random(seed)
    pools = {key: [row for row in rows if stratum(row) == key] for key in STRATA}
    # Reserve scarce distinct contexts first, independently of any model output.
    group_order = sorted(STRATA, key=lambda key: (len({r["context_sha256"] for r in pools[key]}), STRATA.index(key)))
    selected, seen_contexts = [], set()
    per_group = count // 4
    for key in group_order:
        candidates = list(pools[key])
        rng.shuffle(candidates)
        groups = defaultdict(list)
        local_seen = set(seen_contexts)
        for row in candidates:
            if row["context_sha256"] not in local_seen:
                groups[row["article"]].append(row)
                local_seen.add(row["context_sha256"])
        articles = sorted(groups)
        rng.shuffle(articles)
        chosen, depth = [], 0
        while len(chosen) < per_group:
            before = len(chosen)
            for title in articles:
                if depth < len(groups[title]):
                    chosen.append(groups[title][depth])
                    if len(chosen) == per_group:
                        break
            if len(chosen) == before:
                raise ValueError(f"Insufficient distinct passages for {key}: {len(chosen)} of {per_group}.")
            depth += 1
        selected.extend(chosen)
        seen_contexts.update(row["context_sha256"] for row in chosen)
    rng.shuffle(selected)
    return selected


def validate_splits(splits: dict) -> None:
    articles, contexts, identifiers = set(), set(), set()
    for name in ("train", "validation", "test"):
        rows = splits[name]
        counts = Counter(stratum(row) for row in rows)
        if not rows or set(counts) != set(STRATA) or len(set(counts.values())) != 1:
            raise ValueError("Each split must have equal nonzero counts in all four strata.")
        current_articles = {row["article_sha256"] for row in rows}
        current_contexts = {row["context_sha256"] for row in rows}
        current_ids = {row["id"] for row in rows}
        if len(current_contexts) != len(rows) or len(current_ids) != len(rows):
            raise ValueError("Only one question per distinct passage is allowed within a split.")
        if articles & current_articles or contexts & current_contexts or identifiers & current_ids:
            raise ValueError("Article, question or context leakage across splits.")
        for row in rows:
            if row["is_impossible"]:
                if row["references"] or row["response"] != MARKER:
                    raise ValueError("Impossible rows require an empty reference list and NOT_FOUND target.")
            elif not row["references"] or row["response"] != row["references"][0] or any(a.strip() == MARKER for a in row["references"]):
                raise ValueError("Answerable rows require a non-marker reference answer target.")
        articles.update(current_articles)
        contexts.update(current_contexts)
        identifiers.update(current_ids)


def select_splits(train_rows: list[dict], dev_rows: list[dict], config: dict, controls: dict) -> tuple[dict, dict]:
    train_articles = {r["article_sha256"] for r in train_rows}
    dev_articles = {r["article_sha256"] for r in dev_rows}
    if train_articles & dev_articles:
        raise ValueError("Official source article overlap is not allowed.")
    old_articles = set(controls["prior_train_and_validation_article_sha256"])
    validation_articles = train_articles - old_articles
    if not validation_articles:
        raise ValueError("No validation articles unseen by the previous experiment remain.")
    shared_contexts = {r["context_sha256"] for r in train_rows} & {r["context_sha256"] for r in dev_rows}
    prior_ids = set(controls["prior_test_ids"])
    prior_contexts = set(controls["prior_test_context_sha256"])
    test_excluded = [r for r in dev_rows if r["id"] in prior_ids or r["context_sha256"] in prior_contexts]
    pools = {
        "train": [r for r in train_rows if r["article_sha256"] not in validation_articles and r["context_sha256"] not in shared_contexts],
        "validation": [r for r in train_rows if r["article_sha256"] in validation_articles and r["context_sha256"] not in shared_contexts],
        "test": [r for r in dev_rows if r["id"] not in prior_ids and r["context_sha256"] not in prior_contexts and r["context_sha256"] not in shared_contexts],
    }
    splits = {name: sample_stratified(pools[name], config[f"{name}_size"], config["seed"] + offset + 1)
              for offset, name in enumerate(("train", "validation", "test"))}
    validate_splits(splits)
    summary = {
        "validation_article_pool_sha256": sorted(validation_articles),
        "validation_article_pool_count": len(validation_articles),
        "eligible_pool_sizes": {name: len(rows) for name, rows in pools.items()},
        "eligible_pool_stratum_counts": {name: dict(Counter(stratum(r) for r in rows)) for name, rows in pools.items()},
        "selected_stratum_counts": {name: dict(Counter(stratum(r) for r in rows)) for name, rows in splits.items()},
        "selected_article_counts": {name: len({r["article_sha256"] for r in rows}) for name, rows in splits.items()},
        "exact_cross_source_duplicate_contexts_removed": len(shared_contexts),
        "prior_test_rows_excluded_by_id_or_context": len(test_excluded),
        "prior_test_contexts_excluded": len({r["context_sha256"] for r in test_excluded}),
        "selected_test_articles_previously_evaluated": len({r["article_sha256"] for r in splits["test"]} & set(controls["prior_test_article_sha256"])),
    }
    return splits, summary


def public_manifest(splits: dict) -> dict:
    return {name: [{"id": row["id"], "source": row["source"], "article_sha256": row["article_sha256"],
                    "context_sha256": row["context_sha256"], "question_sha256": row["question_sha256"],
                    "reference_sha256": [sha256(text.encode("utf-8")) for text in row["references"]],
                    "response_sha256": sha256(row["response"].encode("utf-8")),
                    "is_impossible": row["is_impossible"], "length_group": row["length_group"],
                    "prompt_tokens": row["prompt_tokens"], "context_tokens": row["context_tokens"],
                    "reference_token_counts": row["reference_token_counts"]} for row in rows] for name, rows in splits.items()}


def prepare(config: dict, output: Path, cache: Path, download: bool, prior_manifest: Path) -> tuple[dict, dict]:
    from transformers import AutoTokenizer
    validate_config(config)
    if any((output / name).exists() for name in ("data", "split_manifest.json", "data_provenance.json")):
        raise FileExistsError("Previous prepared data is preserved; choose a new output directory.")
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["revision"], local_files_only=True, trust_remote_code=False)
    prior_payload = prior_manifest.read_bytes()
    controls = prior_controls(json.loads(prior_payload))
    sources = read_sources(cache, download)
    train, train_stats = eligible_records(sources["train-v2.0.json"], tokenizer, config, "official_train")
    dev, dev_stats = eligible_records(sources["dev-v2.0.json"], tokenizer, config, "official_dev_reserved_as_local_test")
    splits, selection = select_splits(train, dev, config, controls)
    data = output / "data"
    data.mkdir(parents=True)
    hashes = {}
    for name, rows in splits.items():
        path = data / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")
        hashes[name] = sha256(path.read_bytes())
    write_json(output / "split_manifest.json", public_manifest(splits))
    provenance = {
        "dataset": "SQuAD2.0 balanced answerability and short/long complete-passage subset",
        "source_revision": SOURCE_REVISION, "source_urls": {name: SOURCE_BASE + "/" + name for name in SOURCES}, "source_sha256": SOURCES,
        "citation": "Rajpurkar, Jia, Liang (2018). Know What You Don't Know: Unanswerable Questions for SQuAD. arXiv:1806.03822.",
        "license": "CC BY-SA 4.0", "license_source": "https://rajpurkar.github.io/SQuAD-explorer/",
        "data_config": {key: config[key] for key in DATA_KEYS},
        "eligibility": {"official_train": train_stats, "official_dev": dev_stats},
        "prior_manifest_sha256": sha256(prior_payload), "prior_experiment_controls": controls,
        "selection": selection, "counts": {name: len(rows) for name, rows in splits.items()},
        "local_data_sha256": hashes, "split_manifest_sha256": sha256((output / "split_manifest.json").read_bytes()),
        "policy": "All four answerability/length strata equally represented; one question per distinct context across each split. Complete original passages only; no cropping, answer removal, distractor insertion or truncation. Native impossible questions have no gold answers; plausible_answers are never targets. All official-train articles unseen in both prior train and validation form the new validation pool. New test excludes prior test IDs and contexts; its source articles were evaluated previously. Current train/validation/test articles are disjoint. Gold answers filter eligibility and scoring but never construct model input.",
        "sampling": "Rarest distinct-context stratum first, then seeded shuffled questions with round-robin shuffled articles; reserve selected contexts globally before the next stratum.",
        "limitations": ["Balanced filtered subset, not the full SQuAD2 distribution or hidden official test.",
                        f"Long means {config['short_context_tokens'] + 1}-{config['max_context_tokens']} model tokens under the recorded configuration, not arbitrary multi-page documents.",
                        "New test records differ from the previous test, but the Wikipedia articles and some question forms are related.",
                        "Historical public material may have appeared in base-model pretraining; exact checks cannot rule out near duplicates or related facts."],
    }
    write_json(output / "data_provenance.json", provenance)
    return splits, provenance


def load_prepared(directory: Path, config: dict) -> tuple[dict, dict]:
    validate_config(config)
    provenance = json.loads((directory / "data_provenance.json").read_bytes())
    if provenance["data_config"] != {key: config[key] for key in DATA_KEYS}:
        raise ValueError("Prepared dataset configuration differs from this run.")
    manifest_payload = (directory / "split_manifest.json").read_bytes()
    if sha256(manifest_payload) != provenance["split_manifest_sha256"]:
        raise ValueError("Prepared split manifest checksum mismatch.")
    splits = {}
    for name in ("train", "validation", "test"):
        path = directory / "data" / f"{name}.jsonl"
        if sha256(path.read_bytes()) != provenance["local_data_sha256"][name]:
            raise ValueError("Prepared local data checksum mismatch.")
        splits[name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        if len(splits[name]) != config[f"{name}_size"]:
            raise ValueError("Prepared split count differs from configuration.")
    validate_splits(splits)
    if public_manifest(splits) != json.loads(manifest_payload):
        raise ValueError("Prepared rows do not match their public manifest.")
    controls = provenance["prior_experiment_controls"]
    previous_articles = set(controls["prior_train_and_validation_article_sha256"])
    if any(row["article_sha256"] in previous_articles for row in splits["validation"] + splits["test"]):
        raise ValueError("Held-out articles overlap the previous adaptation/validation material.")
    if any(row["id"] in set(controls["prior_test_ids"]) or row["context_sha256"] in set(controls["prior_test_context_sha256"]) for row in splits["test"]):
        raise ValueError("New test repeats a previous test question or context.")
    for rows in splits.values():
        for row in rows:
            group = "short" if row["context_tokens"] <= config["short_context_tokens"] else "long"
            if row["length_group"] != group or not 0 < row["context_tokens"] <= config["max_context_tokens"] or row["prompt_tokens"] > config["max_prompt_tokens"]:
                raise ValueError("Prepared length metadata exceeds configured limits.")
    return splits, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/document-v2.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache/document-v2-data"))
    parser.add_argument("--prior-manifest", type=Path, default=Path("results/document-split-manifest.json"))
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--prepare-only", action="store_true", help="Accepted for symmetry; this command only prepares data.")
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Choose a new output directory.")
    config = json.loads(args.config.read_bytes())
    _, provenance = prepare(config, args.output, args.cache, args.download_data, args.prior_manifest)
    write_json(args.output / "resolved_config.json", config)
    write_json(args.output / "run_status.json", {"status": "prepared"})
    print(json.dumps({"counts": provenance["counts"], "selection": provenance["selection"], "eligibility": provenance["eligibility"]}, indent=2))


if __name__ == "__main__":
    main()
