"""Fresh, complete DuoRC ParaphraseRC plots for a CPU extractive reader.

Native no-answer labels are retained. Literal character offsets are derived
only for positive supervision; the complete original answer list remains the
scoring reference. No passage cropping or synthetic missing answers are used.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
import hashlib
import json
from pathlib import Path
import random
import urllib.request

from .document_v2_metrics import normalize_answer

MARKER = "NOT_FOUND"
PROMPT_INSTRUCTION = ("Answer the question using the passage. Copy only the shortest answer span. "
                      "If the passage does not contain the answer, output exactly NOT_FOUND. "
                      "Do not add explanations.\nPassage:\n")
STRATA = ("answerable_short", "answerable_long", "missing_short", "missing_long")
SPLITS = ("train", "validation", "test")
SOURCE_REVISION = "f8520942ea5e98e135e1eae0f6851db86d3407d9"
SOURCE_BASE = f"https://raw.githubusercontent.com/duorc/duorc/{SOURCE_REVISION}"
SOURCES = {
    "ParaphraseRC_train.json": "8a22500def833cf33a4eaaf7c00b5fc144b5fe26d40e6d94a96bd23a88056193",
    "ParaphraseRC_dev.json": "6d49a1ace1cebe7c73063551e91548725c75db505e1ada58af81c0e1dd21fe60",
    "ParaphraseRC_test.json": "98e471e3b8203664b1656f19fd22d50c3e8ec4e84792846ccf1ccbf0c1298799",
}
LICENSE_SHA256 = "d09d26b69db6c9cabcc9c6bc316e54a5bb50ec7f328b44f0713d621008622af1"
DATA_KEYS = (
    "model_id", "revision", "reader_model_id", "reader_revision", "seed",
    "train_size", "validation_size", "test_size", "short_context_tokens",
    "max_context_tokens", "max_prompt_tokens", "max_answer_tokens",
    "reader_max_length", "reader_max_answer_tokens", "max_questions_per_movie",
)


def sha256(value: bytes | str) -> str:
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def prompt_text(context: str, question: str) -> str:
    return PROMPT_INSTRUCTION + context + "\nQuestion: " + question


def stratum(row: dict) -> str:
    return ("missing_" if row["is_impossible"] else "answerable_") + row["length_group"]


def validate_config(config: dict) -> None:
    for key in DATA_KEYS:
        if key not in config:
            raise ValueError(f"Missing reader dataset setting: {key}")
    for key in DATA_KEYS[4:]:
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Reader dataset setting must be a positive integer: {key}")
    for key in DATA_KEYS[:4]:
        if not isinstance(config[key], str) or not config[key]:
            raise ValueError(f"Reader dataset model/revision must be nonempty: {key}")
    if config["short_context_tokens"] >= config["max_context_tokens"]:
        raise ValueError("Short context boundary must be below the maximum.")
    if any(config[name + "_size"] % 2 for name in SPLITS):
        raise ValueError("Every split must have an even size for balanced answerability.")


def read_sources(cache: Path, download: bool) -> dict:
    sources = {}
    for name, expected in {**SOURCES, "LICENSE": LICENSE_SHA256}.items():
        path = cache / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing pinned DuoRC file: {name}")
            url = SOURCE_BASE + ("/LICENSE" if name == "LICENSE" else "/dataset/" + name)
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read(60_000_001)
            if len(payload) > 60_000_000 or sha256(payload) != expected:
                raise ValueError(f"Pinned DuoRC download checksum/size mismatch: {name}")
            cache.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
        payload = path.read_bytes()
        if sha256(payload) != expected:
            raise ValueError(f"Pinned DuoRC checksum mismatch: {name}")
        if name != "LICENSE":
            sources[name] = json.loads(payload)
    return sources


def exclusion_controls(manifests: list[dict], hashes: list[str]) -> dict:
    if not manifests or len(manifests) != len(hashes) or len(set(hashes)) != len(hashes):
        raise ValueError("Distinct ordered prior manifests and checksums are required.")
    for manifest in manifests:
        if any(not isinstance(manifest.get(name), list) or not manifest[name] for name in SPLITS):
            raise ValueError("Every historical manifest must contain three nonempty splits.")
        if any(not all(key in row for key in ("id", "context_sha256")) for name in SPLITS for row in manifest[name]):
            raise ValueError("Historical rows need question IDs and context hashes.")
    records = [row for manifest in manifests for name in SPLITS for row in manifest[name]]
    return {
        "prior_manifests": [{"index": index, "sha256": digest, "split_counts": {name: len(manifest[name]) for name in SPLITS}}
                            for index, (manifest, digest) in enumerate(zip(manifests, hashes, strict=True))],
        "all_prior_ids": sorted({row["id"] for row in records}),
        "all_prior_context_sha256": sorted({row["context_sha256"] for row in records}),
        "prior_record_count_with_repetition": len(records),
    }


def aligned_literal_spans(context, references, generator, encoded, config):
    """Derive exact offsets; no normalized or approximate answer matching."""
    offsets, sequences = encoded["offset_mapping"], encoded.sequence_ids()
    starts = {value[0]: index for index, value in enumerate(offsets) if sequences[index] == 1 and value[1] > value[0]}
    ends = {value[1]: index for index, value in enumerate(offsets) if sequences[index] == 1 and value[1] > value[0]}
    spans = set()
    for answer in references:
        if not answer or len(generator.encode(answer, add_special_tokens=False)) > config["max_answer_tokens"]:
            continue
        start = context.find(answer)
        while start >= 0:
            end = start + len(answer)
            if start in starts and end in ends and 0 <= ends[end] - starts[start] < config["reader_max_answer_tokens"]:
                spans.add((start, end, answer))
            start = context.find(answer, start + 1)
    return [{"text": text, "start": start, "end": end} for start, end, text in sorted(spans)]


def eligible_records(documents, generator, reader, config, source, controls):
    stats, records = Counter(), []
    old_ids, old_contexts = set(controls["all_prior_ids"]), set(controls["all_prior_context_sha256"])
    seen_ids, seen_questions = set(), set()
    for movie in documents:
        movie_id, context = movie["id"], movie["plot"]
        context_hash = sha256(context)
        context_tokens = len(generator.encode(context, add_special_tokens=False))
        stats["source_movies"] += 1
        for question in movie["qa"]:
            stats["source_questions"] += 1
            source_id = question["id"]
            identifier = "duorc:ParaphraseRC:" + source_id
            if identifier in seen_ids:
                raise ValueError("Native DuoRC question IDs must be unique within each split.")
            seen_ids.add(identifier)
            missing, references = question["no_answer"], question["answers"]
            if type(missing) is not bool or not isinstance(references, list) or any(not isinstance(value, str) for value in references):
                raise ValueError("DuoRC requires native boolean labels and string reference lists.")
            if missing == bool(references):
                raise ValueError("DuoRC native no-answer label disagrees with its references.")
            if identifier in old_ids or source_id in old_ids or context_hash in old_contexts:
                stats["rejected_historical_id_or_context"] += 1
                continue
            if not 0 < context_tokens <= config["max_context_tokens"]:
                stats["rejected_complete_context_length"] += 1
                continue
            if any(not normalize_answer(value) or value.strip() == MARKER for value in references):
                stats["rejected_empty_or_marker_reference"] += 1
                continue
            prompt = prompt_text(context, question["question"])
            rendered = generator.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False, add_generation_prompt=True) if generator.chat_template else "User: " + prompt + "\nAssistant: "
            prompt_tokens = len(generator.encode(rendered, add_special_tokens=False))
            if prompt_tokens > config["max_prompt_tokens"]:
                stats["rejected_generator_prompt_length"] += 1
                continue
            encoded = reader(question["question"], context, return_offsets_mapping=True, truncation=False)
            if len(encoded["input_ids"]) > config["reader_max_length"]:
                stats["rejected_reader_joint_length"] += 1
                continue
            spans = [] if missing else aligned_literal_spans(context, references, generator, encoded, config)
            if not missing and not spans:
                stats["rejected_no_bounded_aligned_literal_span"] += 1
                continue
            question_hash = sha256(question["question"])
            signature = (movie_id, question_hash)
            if signature in seen_questions:
                stats["rejected_duplicate_question_text_within_movie"] += 1
                continue
            seen_questions.add(signature)
            response = MARKER if missing else spans[0]["text"]
            row = {
                "id": identifier, "source_question_id": source_id, "source": source,
                "movie_id": movie_id, "article": movie["title"],
                "movie_sha256": sha256(movie_id), "article_sha256": sha256(movie_id),
                "context": context, "question": question["question"], "prompt": prompt,
                "references": list(references), "response": response, "is_impossible": missing,
                "literal_answer_spans": spans, "context_sha256": context_hash,
                "question_sha256": question_hash,
                "context_tokens": context_tokens, "prompt_tokens": prompt_tokens,
                "reader_tokens": len(encoded["input_ids"]),
                "target_generator_tokens": len(generator.encode(response, add_special_tokens=False)),
                "reference_token_counts": [len(generator.encode(value, add_special_tokens=False)) for value in references],
                "length_group": "short" if context_tokens <= config["short_context_tokens"] else "long",
            }
            records.append(row)
            stats["eligible_" + stratum(row)] += 1
    stats["eligible_questions"] = len(records)
    stats["eligible_movies"] = len({row["movie_sha256"] for row in records})
    stats["eligible_contexts"] = len({row["context_sha256"] for row in records})
    return records, dict(stats)


def capacity(rows, cap):
    counts = defaultdict(Counter)
    for row in rows:
        counts[row["movie_sha256"]][row["is_impossible"]] += 1
    return {"answerable_capacity": sum(min(cap, value[False]) for value in counts.values()),
            "missing_capacity": sum(min(cap, value[True]) for value in counts.values()),
            "total_capacity": sum(min(cap, sum(value.values())) for value in counts.values()),
            "eligible_movies": len(counts)}


def sample_balanced(rows, size, cap, seed):
    """Seeded integer max-flow prevents class-first greedy allocation failure."""
    if size < 2 or size % 2 or cap < 1:
        raise ValueError("Balanced sampling needs an even size and positive movie cap.")
    available = capacity(rows, cap)
    if min(available["answerable_capacity"], available["missing_capacity"]) < size // 2 or available["total_capacity"] < size:
        raise ValueError("Insufficient eligible questions under the balanced movie cap.")
    rng = random.Random(seed)
    grouped = defaultdict(lambda: {False: [], True: []})
    for row in rows:
        grouped[row["movie_sha256"]][row["is_impossible"]].append(row)
    movies = sorted(grouped)
    rng.shuffle(movies)
    for movie in movies:
        for label in (False, True):
            rng.shuffle(grouped[movie][label])
    # Nodes: source=0, answerable=1, missing=2, one per movie, sink.
    sink = len(movies) + 3
    graph = [[] for _ in range(sink + 1)]

    def edge(first, second, amount):
        forward = [second, amount, len(graph[second])]
        graph[first].append(forward)
        graph[second].append([first, 0, len(graph[first]) - 1])
        return forward

    edge(0, 1, size // 2)
    edge(0, 2, size // 2)
    allocations = []
    for index, movie in enumerate(movies):
        node = index + 3
        edge(node, sink, cap)
        for label in (False, True):
            amount = min(cap, len(grouped[movie][label]))
            reference = edge(2 if label else 1, node, amount)
            allocations.append((movie, label, amount, reference))
    flow = 0
    while flow < size:
        previous, queue = {0: None}, deque([0])
        while queue and sink not in previous:
            node = queue.popleft()
            for index, (destination, remaining, _) in enumerate(graph[node]):
                if remaining > 0 and destination not in previous:
                    previous[destination] = (node, index)
                    queue.append(destination)
        if sink not in previous:
            raise ValueError("Insufficient balanced allocation after movie constraints.")
        amount, node = size - flow, sink
        while node:
            parent, index = previous[node]
            amount = min(amount, graph[parent][index][1])
            node = parent
        node = sink
        while node:
            parent, index = previous[node]
            connection = graph[parent][index]
            connection[1] -= amount
            graph[node][connection[2]][1] += amount
            node = parent
        flow += amount
    selected = []
    for movie, label, original, connection in allocations:
        selected.extend(grouped[movie][label][:original - connection[1]])
    rng.shuffle(selected)
    return selected


def validate_splits(splits: dict, config: dict) -> None:
    seen_movies, seen_contexts, seen_ids = set(), set(), set()
    for name in SPLITS:
        rows = splits[name]
        if len(rows) != config[name + "_size"] or sum(row["is_impossible"] for row in rows) * 2 != len(rows):
            raise ValueError("Selected split size/answerability differs from the configured balance.")
        ids, contexts, movies = ({row[key] for row in rows} for key in ("id", "context_sha256", "movie_sha256"))
        if len(ids) != len(rows) or ids & seen_ids or contexts & seen_contexts or movies & seen_movies:
            raise ValueError("Question, complete context or movie leakage across splits.")
        if max(Counter(row["movie_sha256"] for row in rows).values()) > config["max_questions_per_movie"]:
            raise ValueError("Too many selected questions from one movie.")
        if len({(row["movie_sha256"], row["question_sha256"]) for row in rows}) != len(rows):
            raise ValueError("Selected movie questions must have distinct text.")
        for row in rows:
            if type(row["is_impossible"]) is not bool:
                raise ValueError("Native answerability must be boolean.")
            if row["id"] != "duorc:ParaphraseRC:" + row["source_question_id"]:
                raise ValueError("DuoRC question namespace differs.")
            if row["article_sha256"] != row["movie_sha256"] or row["movie_sha256"] != sha256(row["movie_id"]):
                raise ValueError("Movie identity checksum differs.")
            if row["context_sha256"] != sha256(row["context"]) or row["question_sha256"] != sha256(row["question"]):
                raise ValueError("Context/question checksum differs.")
            if row["prompt"] != prompt_text(row["context"], row["question"]):
                raise ValueError("Prompt must contain exactly the full context and question.")
            if not 0 < row["context_tokens"] <= config["max_context_tokens"] or not 0 < row["prompt_tokens"] <= config["max_prompt_tokens"] or not 0 < row["reader_tokens"] <= config["reader_max_length"]:
                raise ValueError("Recorded complete input exceeds token limits.")
            if row["length_group"] != ("short" if row["context_tokens"] <= config["short_context_tokens"] else "long"):
                raise ValueError("Recorded length group differs.")
            if not 0 < row["target_generator_tokens"] <= config["max_answer_tokens"]:
                raise ValueError("Selected target exceeds the generator answer limit.")
            spans, references = row["literal_answer_spans"], row["references"]
            if row["is_impossible"]:
                if spans or references or row["response"] != MARKER:
                    raise ValueError("Native missing answers need no references or spans and target NOT_FOUND.")
            elif not references or not spans or row["response"] != spans[0]["text"]:
                raise ValueError("Positive examples need an eligible literal target and original references.")
            if any(not normalize_answer(value) or value.strip() == MARKER for value in references):
                raise ValueError("Gold reference is empty or conflicts with the marker.")
            for span in spans:
                if type(span["start"]) is not int or type(span["end"]) is not int or not 0 <= span["start"] < span["end"] <= len(row["context"]):
                    raise ValueError("Invalid literal span offsets.")
                if row["context"][span["start"]:span["end"]] != span["text"] or span["text"] not in references:
                    raise ValueError("Literal target is not an unchanged original gold answer span.")
        seen_ids.update(ids)
        seen_contexts.update(contexts)
        seen_movies.update(movies)


def check_historical_exclusions(splits: dict, controls: dict) -> None:
    identifiers, contexts = set(controls["all_prior_ids"]), set(controls["all_prior_context_sha256"])
    for rows in splits.values():
        if any(row["id"] in identifiers or row["source_question_id"] in identifiers or row["context_sha256"] in contexts for row in rows):
            raise ValueError("Selected data repeats a historical question ID or context.")


def public_manifest(splits: dict) -> dict:
    fields = ("id", "source", "movie_sha256", "article_sha256", "context_sha256", "question_sha256",
              "is_impossible", "length_group", "context_tokens", "prompt_tokens", "reader_tokens",
              "target_generator_tokens", "reference_token_counts")
    return {name: [{**{key: row[key] for key in fields},
                    "reference_sha256": [sha256(value) for value in row["references"]],
                    "response_sha256": sha256(row["response"]),
                    "literal_answer_spans": [{"text_sha256": sha256(span["text"]), "start": span["start"], "end": span["end"]} for span in row["literal_answer_spans"]]}
                   for row in rows] for name, rows in splits.items()}


def prepare(config: dict, output: Path, cache: Path, download: bool, prior_manifests: list[Path]):
    validate_config(config)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Existing reader data is preserved; choose an empty output directory.")
    payloads = [Path(path).read_bytes() for path in prior_manifests]
    controls = exclusion_controls([json.loads(value) for value in payloads], [sha256(value) for value in payloads])
    from transformers import AutoTokenizer
    generator = AutoTokenizer.from_pretrained(config["model_id"], revision=config["revision"], local_files_only=True, trust_remote_code=False)
    reader = AutoTokenizer.from_pretrained(config["reader_model_id"], revision=config["reader_revision"], local_files_only=True, trust_remote_code=False, use_fast=True)
    if not reader.is_fast:
        raise ValueError("Reader requires fast-tokenizer character offsets.")
    sources = read_sources(cache, download)
    splits, eligibility, availability = {}, {}, {}
    all_movies, all_contexts, all_ids = set(), set(), set()
    for index, (name, filename) in enumerate(zip(SPLITS, SOURCES, strict=True)):
        documents = sources[filename]
        movies = {movie["id"] for movie in documents}
        contexts = {sha256(movie["plot"]) for movie in documents}
        identifiers = {question["id"] for movie in documents for question in movie["qa"]}
        if movies & all_movies or contexts & all_contexts or identifiers & all_ids:
            raise ValueError("Official DuoRC splits overlap by movie, full context or question ID.")
        all_movies.update(movies)
        all_contexts.update(contexts)
        all_ids.update(identifiers)
        rows, eligibility[name] = eligible_records(documents, generator, reader, config, filename, controls)
        availability[name] = capacity(rows, config["max_questions_per_movie"])
        splits[name] = sample_balanced(rows, config[name + "_size"], config["max_questions_per_movie"], config["seed"] + index)
    validate_splits(splits, config)
    check_historical_exclusions(splits, controls)
    (output / "data").mkdir(parents=True)
    local_hashes = {}
    for name, rows in splits.items():
        path = output / "data" / (name + ".jsonl")
        path.write_text("".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows), encoding="utf-8", newline="\n")
        local_hashes[name] = sha256(path.read_bytes())
    write_json(output / "split_manifest.json", public_manifest(splits))
    provenance = {
        "dataset": "DuoRC ParaphraseRC complete-plot extractive subset with native no-answer labels",
        "source_revision": SOURCE_REVISION, "source_sha256": SOURCES,
        "source_urls": {name: SOURCE_BASE + "/dataset/" + name for name in SOURCES},
        "citation": "Saha, Aralikatte, Khapra, Sankaranarayanan (2018). DuoRC: Towards Complex Language Understanding with Paraphrased Reading Comprehension. ACL. arXiv:1804.07927.",
        "license": "MIT (authors' dataset repository)", "license_source": SOURCE_BASE + "/LICENSE", "license_sha256": LICENSE_SHA256,
        "data_config": {key: config[key] for key in DATA_KEYS},
        "prior_manifest_sha256": [entry["sha256"] for entry in controls["prior_manifests"]],
        "historical_exclusion_controls": controls, "eligibility": eligibility,
        "availability_before_generation": availability,
        "counts": {name: len(rows) for name, rows in splits.items()},
        "selection": {name: {"by_stratum": dict(Counter(stratum(row) for row in rows)),
                             "movies": len({row["movie_sha256"] for row in rows}),
                             "contexts": len({row["context_sha256"] for row in rows}),
                             "max_questions_per_movie": max(Counter(row["movie_sha256"] for row in rows).values())}
                      for name, rows in splits.items()},
        "local_data_sha256": local_hashes,
        "split_manifest_sha256": sha256((output / "split_manifest.json").read_bytes()),
        "policy": "Official train/dev/test map to local train/validation/test. Movie IDs, exact complete contexts and question IDs are disjoint across splits. All historical manifest IDs and contexts are excluded from every split. Native no_answer is preserved; negative targets are NOT_FOUND with empty references. Positive supervision derives literal exact offsets with both tokenizer budgets and exact reader token boundaries. Full original answer lists remain scoring references. Identical question text within one movie is deduplicated; at most four distinct questions per movie are selected by a seeded balanced integer-flow allocation. No model outputs influence eligibility or sampling.",
        "balance": "Exactly half answerable and half native missing-answer examples. Short/long groups are descriptive, not equally weighted; complete ParaphraseRC plots contain too few short answerable questions for the earlier four-way quota.",
        "span_policy": "Offsets are derived from exact source gold strings, not supplied natively by DuoRC. All eligible exact occurrences are stored; the earliest start/end/text is the deterministic training target. A source answer with no eligible literal span is excluded, never relabeled missing. Original answers, complete context and question remain unchanged.",
        "sampling": "Seeded shuffle of sorted movie IDs and each movie/class question pool; integer max-flow from balanced class quotas through per-movie capacities, followed by a seeded selected-row shuffle. Split seeds are seed, seed+1, seed+2.",
        "limitations": ["Filtered historical movie-plot corpus, not native answerability prevalence or unseen real-world documents.",
                        "Multiple questions share a movie context within a split; uncertainty should resample movies as clusters.",
                        "The QA reader was initialized from SQuAD2 training. Base-model pretraining exposure and cross-corpus near duplicates cannot be ruled out.",
                        "DuoRC includes paraphrased/abstractive answers. Requiring one short literal span narrows the task; full original answer lists are still used for scoring.",
                        "Repository MIT notice is recorded; source plots originate from IMDb and Wikipedia. Raw plots/questions/references stay in ignored local artifacts.",
                        "Complete contexts must fit both models. No cropping, sliding windows, answer deletion, generated negatives or model-based filtering is used."],
    }
    write_json(output / "data_provenance.json", provenance)
    return splits, provenance


def load_prepared(output: Path, config: dict):
    validate_config(config)
    provenance = json.loads((output / "data_provenance.json").read_bytes())
    if provenance["data_config"] != {key: config[key] for key in DATA_KEYS}:
        raise ValueError("Prepared reader dataset configuration differs.")
    controls = provenance["historical_exclusion_controls"]
    if provenance["prior_manifest_sha256"] != [entry["sha256"] for entry in controls["prior_manifests"]]:
        raise ValueError("Historical manifest checksum bindings disagree.")
    manifest = (output / "split_manifest.json").read_bytes()
    if sha256(manifest) != provenance["split_manifest_sha256"]:
        raise ValueError("Prepared reader split manifest checksum mismatch.")
    splits = {}
    for name in SPLITS:
        payload = (output / "data" / (name + ".jsonl")).read_bytes()
        if sha256(payload) != provenance["local_data_sha256"][name]:
            raise ValueError("Prepared reader local data checksum mismatch.")
        splits[name] = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    validate_splits(splits, config)
    check_historical_exclusions(splits, controls)
    if public_manifest(splits) != json.loads(manifest):
        raise ValueError("Prepared reader rows differ from their public manifest.")
    return splits, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/reader.json"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path, default=Path(".cache/duorc-research"))
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prior-manifests", type=Path, nargs="+", default=[Path("results/document-split-manifest.json"), Path("results/document-v2-split-manifest.json"), Path("results/document-verifier/split_manifest.json")])
    args = parser.parse_args()
    config = json.loads(args.config.read_bytes())
    _, provenance = prepare(config, args.output, args.cache, args.download_data, args.prior_manifests)
    write_json(args.output / "resolved_config.json", config)
    write_json(args.output / "run_status.json", {"status": "prepared"})
    print(json.dumps({"counts": provenance["counts"], "selection": provenance["selection"], "manifest_sha256": provenance["split_manifest_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
