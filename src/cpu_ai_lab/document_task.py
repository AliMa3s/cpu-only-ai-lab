"""CPU LoRA answer extraction from complete, short SQuAD Wikipedia passages.

This is free answer generation, not constrained label classification. Published
metrics contain IDs, hashes and scores; passages and answer strings stay local.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import re
import string
import time
import urllib.request

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from .training import PeakRSS, adapter_snapshot, assert_cpu_model, collate_batch, encode_examples, render_prompt, write_json

SOURCE_REVISION = "e0c66cfec263165fb3c15cb76a22175e371b71d7"
SOURCE_BASE = f"https://raw.githubusercontent.com/rajpurkar/SQuAD-explorer/{SOURCE_REVISION}/dataset"
SOURCES = {
    "train-v1.1.json": "3527663986b8295af4f7fcdff1ba1ff3f72d07d61a20f487cb238a6ef92fd955",
    "dev-v1.1.json": "95aa6a52d5d6a735563366753ca50492a658031da74f301ac5238b03966972c9",
}
DATA_KEYS = ("model_id", "revision", "seed", "train_size", "validation_size", "test_size", "validation_articles",
             "max_context_tokens", "max_prompt_tokens", "max_answer_tokens", "max_sequence_length")
INSTRUCTION = "Answer the question using the passage. Copy only the shortest answer span. Do not add explanations.\nPassage:\n"


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def directory_hashes(directory: Path) -> dict:
    return {path.name: sha256(path.read_bytes()) for path in sorted(directory.iterdir()) if path.is_file()}


def verify_saved_artifacts(directory: Path, metrics: dict) -> None:
    for filename, key in (("resolved_config.json", "resolved_config_sha256"),
                          ("training_curve.json", "training_curve_sha256"),
                          ("data_provenance.json", "data_provenance_sha256")):
        if sha256((directory / filename).read_bytes()) != metrics[key]:
            raise ValueError(f"Saved {filename} checksum mismatch.")
    if sha256((directory / "split_manifest.json").read_bytes()) != metrics["dataset"]["split_manifest_sha256"]:
        raise ValueError("Saved split manifest checksum mismatch.")
    for folder, key in (("adapter", "adapter_sha256"), ("tokenizer", "tokenizer_sha256")):
        expected = metrics[key]
        if not expected or any(Path(name).name != name for name in expected):
            raise ValueError(f"Invalid saved {folder} checksum manifest.")
        if directory_hashes(directory / folder) != expected:
            raise ValueError(f"Saved {folder} checksum mismatch.")


def normalize_answer(text: str) -> str:
    text = "".join(char for char in text.lower() if char not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", text).split())


def score_answer(prediction: str, references: list[str]) -> dict:
    if not references:
        raise ValueError("At least one reference answer is required.")
    predicted = normalize_answer(prediction)
    scores = []
    for answer in references:
        expected = normalize_answer(answer)
        common = sum((Counter(predicted.split()) & Counter(expected.split())).values())
        if not predicted or not expected:
            f1 = float(predicted == expected)
        elif not common:
            f1 = 0.0
        else:
            precision, recall = common / len(predicted.split()), common / len(expected.split())
            f1 = 2 * precision * recall / (precision + recall)
        scores.append({"exact_match": float(predicted == expected), "token_f1": f1})
    return {"exact_match": max(row["exact_match"] for row in scores), "token_f1": max(row["token_f1"] for row in scores)}


def prompt_text(context: str, question: str) -> str:
    return INSTRUCTION + context + "\nQuestion: " + question


def read_sources(cache: Path, download: bool) -> dict:
    documents = {}
    for name, expected in SOURCES.items():
        path = cache / name
        if not path.exists():
            if not download:
                raise FileNotFoundError(f"Missing {name}; pass --download-data once.")
            cache.mkdir(parents=True, exist_ok=True)
            with urllib.request.urlopen(SOURCE_BASE + "/" + name, timeout=60) as response:
                payload = response.read(40_000_001)
            if len(payload) > 40_000_000 or sha256(payload) != expected:
                raise ValueError("Downloaded SQuAD file failed its pinned checksum/size check.")
            path.write_bytes(payload)
        payload = path.read_bytes()
        if sha256(payload) != expected:
            raise ValueError(f"Pinned SQuAD checksum mismatch: {name}")
        value = json.loads(payload)
        if value.get("version") != "1.1":
            raise ValueError("SQuAD v1.1 is required.")
        documents[name] = value["data"]
    return documents


def eligible_records(documents: list[dict], tokenizer, config: dict, source: str) -> tuple[list[dict], dict]:
    rows, stats = [], Counter()
    context_tokens = {}
    for article in documents:
        title = article["title"]
        for paragraph in article["paragraphs"]:
            context = paragraph["context"]
            context_hash = sha256(context.encode("utf-8"))
            stats["source_paragraphs"] += 1
            if context_hash not in context_tokens:
                context_tokens[context_hash] = len(tokenizer.encode(context, add_special_tokens=False))
            for qa in paragraph["qas"]:
                stats["source_questions"] += 1
                if context_tokens[context_hash] > config["max_context_tokens"]:
                    stats["rejected_context_length"] += 1
                    continue
                references = []
                for answer in qa["answers"]:
                    start, text = answer["answer_start"], answer["text"]
                    if context[start:start + len(text)] != text:
                        raise ValueError("Gold answer is not at its declared source span.")
                    if text not in references:
                        references.append(text)
                lengths = [len(tokenizer.encode(answer, add_special_tokens=False)) for answer in references]
                if not lengths or any(not normalize_answer(answer) for answer in references) or max(lengths) > config["max_answer_tokens"]:
                    stats["rejected_answer_length_or_empty_normalization"] += 1
                    continue
                question = qa["question"]
                prompt = prompt_text(context, question)
                prompt_ids = tokenizer.encode(render_prompt(tokenizer, {"prompt": prompt}), add_special_tokens=False)
                if len(prompt_ids) > config["max_prompt_tokens"]:
                    stats["rejected_prompt_length"] += 1
                    continue
                # The first official reference is the teacher-forced target;
                # every official reference remains eligible for EM/token-F1.
                if len(prompt_ids) + lengths[0] + 1 > config["max_sequence_length"]:
                    raise ValueError("Configured prompt/answer limits do not fit the training sequence budget.")
                rows.append({"id": qa["id"], "article": title, "source": source,
                             "context": context, "question": question, "prompt": prompt, "response": references[0],
                             "references": references, "article_sha256": sha256(title.encode("utf-8")),
                             "context_sha256": context_hash, "question_sha256": sha256(question.encode("utf-8")),
                             "prompt_tokens": len(prompt_ids), "context_tokens": context_tokens[context_hash],
                             "reference_token_counts": lengths})
                stats["eligible_questions"] += 1
    stats["eligible_articles"] = len({row["article"] for row in rows})
    stats["eligible_contexts"] = len({row["context_sha256"] for row in rows})
    return rows, dict(stats)


def sample_articles(rows: list[dict], count: int, seed: int) -> list[dict]:
    """Sample round-robin across shuffled articles, at most one Q per passage."""
    rng = random.Random(seed)
    order = list(rows)
    rng.shuffle(order)
    groups, seen_context = defaultdict(list), set()
    for row in order:
        if row["context_sha256"] not in seen_context:
            groups[row["article"]].append(row)
            seen_context.add(row["context_sha256"])
    articles = sorted(groups)
    rng.shuffle(articles)
    chosen = []
    depth = 0
    while len(chosen) < count:
        previous = len(chosen)
        for title in articles:
            if depth < len(groups[title]):
                chosen.append(groups[title][depth])
                if len(chosen) == count:
                    break
        if len(chosen) == previous:
            raise ValueError(f"Only {len(chosen)} distinct eligible passages available; requested {count}.")
        depth += 1
    rng.shuffle(chosen)
    return chosen


def select_splits(train_rows: list[dict], dev_rows: list[dict], config: dict) -> tuple[dict, dict]:
    train_titles, dev_titles = {r["article"] for r in train_rows}, {r["article"] for r in dev_rows}
    if train_titles & dev_titles:
        raise ValueError("Official source articles overlap; this experiment requires article-disjoint sources.")
    # Remove exact duplicate passages shared across source sets before selection.
    overlapping_contexts = {r["context_sha256"] for r in train_rows} & {r["context_sha256"] for r in dev_rows}
    train_rows = [r for r in train_rows if r["context_sha256"] not in overlapping_contexts]
    dev_rows = [r for r in dev_rows if r["context_sha256"] not in overlapping_contexts]
    titles = sorted({row["article"] for row in train_rows})
    random.Random(config["seed"]).shuffle(titles)
    if not 0 < config["validation_articles"] < len(titles):
        raise ValueError("Validation article count must leave training articles.")
    validation_titles = set(titles[:config["validation_articles"]])
    pools = {"train": [r for r in train_rows if r["article"] not in validation_titles],
             "validation": [r for r in train_rows if r["article"] in validation_titles], "test": dev_rows}
    splits = {name: sample_articles(pools[name], config[f"{name}_size"], config["seed"] + i + 1)
              for i, name in enumerate(("train", "validation", "test"))}
    validate_splits(splits)
    return splits, {"exact_cross_source_duplicate_contexts_removed": len(overlapping_contexts),
                    "eligible_pool_sizes_after_article_split": {k: len(v) for k, v in pools.items()},
                    "selected_article_counts": {k: len({r["article"] for r in v}) for k, v in splits.items()}}


def validate_splits(splits: dict) -> None:
    articles, identifiers, contexts = set(), set(), set()
    for name in ("train", "validation", "test"):
        rows = splits[name]
        if not rows:
            raise ValueError("Empty document split.")
        current_articles = {row["article"] for row in rows}
        current_ids = {row["id"] for row in rows}
        current_contexts = {row["context_sha256"] for row in rows}
        if len(current_ids) != len(rows) or len(current_contexts) != len(rows):
            raise ValueError("Only one question per distinct passage is allowed within a split.")
        if articles & current_articles or identifiers & current_ids or contexts & current_contexts:
            raise ValueError("Article, question, or passage leakage across splits.")
        articles.update(current_articles)
        identifiers.update(current_ids)
        contexts.update(current_contexts)


def public_manifest(splits: dict) -> dict:
    return {name: [{"id": row["id"], "source": row["source"], "article_sha256": row["article_sha256"],
                    "context_sha256": row["context_sha256"], "question_sha256": row["question_sha256"],
                    "reference_sha256": [sha256(answer.encode("utf-8")) for answer in row["references"]],
                    "prompt_tokens": row["prompt_tokens"], "context_tokens": row["context_tokens"],
                    "reference_token_counts": row["reference_token_counts"]} for row in rows] for name, rows in splits.items()}


def prepare(config: dict, output: Path, cache: Path, download: bool) -> tuple[dict, dict]:
    options = {"revision": config["revision"], "local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], **options)
    sources = read_sources(cache, download)
    eligible_train, train_stats = eligible_records(sources["train-v1.1.json"], tokenizer, config, "official_train")
    eligible_dev, dev_stats = eligible_records(sources["dev-v1.1.json"], tokenizer, config, "official_dev_reserved_as_local_test")
    splits, selection = select_splits(eligible_train, eligible_dev, config)
    data = output / "data"
    data.mkdir(parents=True)
    hashes = {}
    for name, rows in splits.items():
        path = data / (name + ".jsonl")
        path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8", newline="\n")
        hashes[name] = sha256(path.read_bytes())
    manifest = public_manifest(splits)
    write_json(output / "split_manifest.json", manifest)
    provenance = {"dataset": "SQuAD v1.1 short-passage, short-answer extraction subset", "source_revision": SOURCE_REVISION,
                  "source_urls": {name: SOURCE_BASE + "/" + name for name in SOURCES}, "source_sha256": SOURCES,
                  "citation": "Rajpurkar, Zhang, Lopyrev, Liang (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text. arXiv:1606.05250.",
                  "license": "CC BY-SA 4.0", "license_source": "https://rajpurkar.github.io/SQuAD-explorer/",
                  "data_config": {k: config[k] for k in DATA_KEYS}, "eligibility": {"official_train": train_stats, "official_dev": dev_stats},
                  "selection": selection, "counts": {k: len(v) for k, v in splits.items()}, "local_data_sha256": hashes,
                  "split_manifest_sha256": sha256((output / "split_manifest.json").read_bytes()),
                  "policy": "Complete passages only, no cropping/truncation. Keep only declared context/prompt limits and questions whose every reference answer fits the target-token limit. First reference is training target; all references score generation. Article-disjoint splits, one question per passage, deterministic round-robin sampling across articles. Official dev is the local held-out test; this is not the official hidden SQuAD test or a full-dataset leaderboard score."}
    write_json(output / "data_provenance.json", provenance)
    return splits, provenance


def load_prepared(directory: Path, config: dict) -> tuple[dict, dict]:
    provenance = json.loads((directory / "data_provenance.json").read_text(encoding="utf-8"))
    if provenance["data_config"] != {k: config[k] for k in DATA_KEYS}:
        raise ValueError("Prepared dataset configuration differs from this run.")
    if sha256((directory / "split_manifest.json").read_bytes()) != provenance["split_manifest_sha256"]:
        raise ValueError("Prepared split manifest checksum mismatch.")
    splits = {}
    for name, expected in provenance["local_data_sha256"].items():
        path = directory / "data" / (name + ".jsonl")
        if sha256(path.read_bytes()) != expected:
            raise ValueError("Prepared local data checksum mismatch.")
        splits[name] = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    validate_splits(splits)
    if public_manifest(splits) != json.loads((directory / "split_manifest.json").read_text(encoding="utf-8")):
        raise ValueError("Prepared rows do not match their public manifest.")
    return splits, provenance


def response_loss(model, batch: dict) -> tuple[torch.Tensor, int]:
    """Exact causal-LM CE with the vocabulary projection only at target positions.

    This path is restricted to Llama causal LMs with ordinary LoRA modules.
    The injected LoRA layers are inside the transformer returned by get_base_model.
    """
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    if getattr(base.config, "model_type", None) != "llama":
        raise ValueError("Response-only projection currently supports Llama causal models only.")
    labels = batch["labels"][:, 1:]
    mask = labels != -100
    count = int(mask.sum())
    if not count:
        raise ValueError("A batch must contain shifted response targets.")
    inputs = {key: value for key, value in batch.items() if key != "labels"}
    hidden = base.model(**inputs, use_cache=False).last_hidden_state[:, :-1, :]
    logits = base.lm_head(hidden[mask]).float()
    return torch.nn.functional.cross_entropy(logits, labels[mask]), count


@torch.inference_mode()
def loss_evaluation(model, encoded: list[dict], pad_id: int, batch_size: int) -> dict:
    model.eval()
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i]["input_ids"]))
    started, nll, targets, allocated, actual = time.perf_counter(), 0.0, 0, 0, 0
    for offset in range(0, len(order), batch_size):
        batch = collate_batch([encoded[i] for i in order[offset:offset + batch_size]], pad_id)
        loss, count = response_loss(model, batch)
        if not torch.isfinite(loss):
            raise RuntimeError("Nonfinite response evaluation loss.")
        nll += float(loss) * count
        targets += count
        allocated += batch["input_ids"].numel()
        actual += int(batch["attention_mask"].sum())
    return {"target_token_loss": nll / targets, "target_tokens": targets, "seconds": time.perf_counter() - started,
            "actual_input_tokens": actual, "allocated_input_tokens_including_padding": allocated, "length_sorted": True}


def left_pad(prompts: list[list[int]], pad_id: int) -> dict:
    width = max(map(len, prompts))
    return {"input_ids": torch.tensor([[pad_id] * (width - len(ids)) + ids for ids in prompts], dtype=torch.long, device="cpu"),
            "attention_mask": torch.tensor([[0] * (width - len(ids)) + [1] * len(ids) for ids in prompts], dtype=torch.long, device="cpu")}


def trim_generated(ids: list[int], eos_id: int) -> list[int]:
    return ids[:ids.index(eos_id) + 1] if eos_id in ids else ids


@torch.inference_mode()
def generation_evaluation(model, tokenizer, rows: list[dict], batch_size: int, token_limit: int) -> tuple[dict, list[dict]]:
    model.eval()
    encoded = [tokenizer.encode(render_prompt(tokenizer, row), add_special_tokens=False) for row in rows]
    order = sorted(range(len(rows)), key=lambda i: len(encoded[i]))
    records, local_predictions = {}, {}
    started = time.perf_counter()
    for offset in range(0, len(order), batch_size):
        indices = order[offset:offset + batch_size]
        batch = left_pad([encoded[i] for i in indices], tokenizer.pad_token_id)
        generated = model.generate(**batch, do_sample=False, use_cache=True, max_new_tokens=token_limit,
                                   eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
        width = batch["input_ids"].shape[1]
        for batch_row, original in enumerate(indices):
            new_ids = trim_generated(generated[batch_row, width:].tolist(), tokenizer.eos_token_id)
            text = tokenizer.decode(new_ids, skip_special_tokens=True)
            source = rows[original]
            scores = score_answer(text, source["references"])
            record = {"id": source["id"], "article_sha256": source["article_sha256"], "context_sha256": source["context_sha256"],
                      "prediction_sha256": sha256(text.encode("utf-8")), "generated_tokens_including_eos": len(new_ids),
                      "ended_with_eos": bool(new_ids and new_ids[-1] == tokenizer.eos_token_id),
                      "verbatim_span_in_passage": bool(text.strip()) and text.strip().casefold() in source["context"].casefold(), **scores}
            records[original] = record
            local_predictions[original] = {**record, "question": source["question"], "references": source["references"], "prediction": text}
        print(json.dumps({"stage": "generation_evaluation", "processed": min(offset + batch_size, len(rows)), "total": len(rows)}), flush=True)
    ordered = [records[i] for i in range(len(rows))]
    metric = {"examples": len(rows), "exact_match": sum(r["exact_match"] for r in ordered) / len(rows),
              "token_f1": sum(r["token_f1"] for r in ordered) / len(rows),
              "verbatim_span_rate": sum(r["verbatim_span_in_passage"] for r in ordered) / len(rows),
              "ended_with_eos": sum(r["ended_with_eos"] for r in ordered),
              "generated_tokens_including_eos": sum(r["generated_tokens_including_eos"] for r in ordered),
              "seconds": time.perf_counter() - started, "batch_size": batch_size, "max_new_tokens": token_limit,
              "length_sorted": True, "records": ordered}
    return metric, [local_predictions[i] for i in range(len(rows))]


def _train(config: dict, output: Path, cache: Path, download: bool, prepared: Path | None):
    started = time.perf_counter()
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("A CPU-only PyTorch build is required.")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["training_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    with PeakRSS() as memory:
        splits, provenance = load_prepared(prepared, config) if prepared else prepare(config, output, cache, download)
        if prepared:
            write_json(output / "data_provenance.json", provenance)
            write_json(output / "split_manifest.json", public_manifest(splits))
        options = {"revision": config["revision"], "local_files_only": True, "trust_remote_code": False}
        tokenizer = AutoTokenizer.from_pretrained(config["model_id"], **options)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(config["model_id"], torch_dtype=torch.float32, attn_implementation="sdpa", **options).cpu()
        base_parameters = sum(p.numel() for p in base.parameters())
        model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=config["lora_rank"], lora_alpha=config["lora_alpha"],
                                              lora_dropout=0.0, target_modules=["q_proj", "v_proj"], bias="none"))
        model.generation_config.do_sample = False
        for key in ("temperature", "top_k", "top_p"):
            setattr(model.generation_config, key, None)
        assert_cpu_model(model)
        initial = adapter_snapshot(model)
        encoded = {name: encode_examples(tokenizer, rows, config["max_sequence_length"]) for name, rows in splits.items()}
        torch.set_num_threads(config["evaluation_threads"])
        print(json.dumps({"stage": "baseline", "test_examples": len(splits["test"])}), flush=True)
        before_validation = loss_evaluation(model, encoded["validation"], tokenizer.pad_token_id, config["loss_batch_size"])
        before_loss = loss_evaluation(model, encoded["test"], tokenizer.pad_token_id, config["loss_batch_size"])
        before_generation, raw_before = generation_evaluation(model, tokenizer, splits["test"], config["generation_batch_size"], config["generation_tokens"])
        write_json(output / "predictions-before-local.json", raw_before)
        before = {"validation": before_validation, "test_loss": before_loss, "test_generation": before_generation}
        write_json(output / "baseline.json", before)
        best_loss, best_step, best_adapter = before_validation["target_token_loss"], 0, initial
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=config["learning_rate"], weight_decay=0.0)
        order = list(range(len(encoded["train"])))
        rng = random.Random(config["seed"])
        rng.shuffle(order)
        position, seen = 0, set()
        curve, compute, target_tokens, input_tokens = [], 0.0, 0, 0
        train_started = time.perf_counter()
        for step in range(1, config["optimizer_steps"] + 1):
            torch.set_num_threads(config["training_threads"])
            indices = []
            for _ in range(config["batch_size"]):
                if position == len(order):
                    rng.shuffle(order)
                    position = 0
                indices.append(order[position])
                position += 1
            seen.update(indices)
            batch = collate_batch([encoded["train"][i] for i in indices], tokenizer.pad_token_id)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            step_started = time.perf_counter()
            loss, count = response_loss(model, batch)
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite response training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0, error_if_nonfinite=True)
            optimizer.step()
            seconds = time.perf_counter() - step_started
            compute += seconds
            target_tokens += count
            input_tokens += int(batch["attention_mask"].sum())
            record = {"step": step, "target_token_loss": float(loss.detach()), "compute_seconds": seconds, "target_tokens": count}
            if step % config["validation_every"] == 0 or step == config["optimizer_steps"]:
                torch.set_num_threads(config["evaluation_threads"])
                validation = loss_evaluation(model, encoded["validation"], tokenizer.pad_token_id, config["loss_batch_size"])
                record["validation"] = validation
                if validation["target_token_loss"] < best_loss:
                    best_loss, best_step, best_adapter = validation["target_token_loss"], step, adapter_snapshot(model)
            curve.append(record)
            write_json(output / "training_curve.json", curve)
            if step % 10 == 0 or step == 1:
                print(json.dumps({"stage": "training", **record}), flush=True)
        loop_seconds = time.perf_counter() - train_started
        set_peft_model_state_dict(model, best_adapter)
        if any(not torch.equal(get_peft_model_state_dict(model)[k].cpu(), v) for k, v in best_adapter.items()):
            raise RuntimeError("Selected adapter restoration failed.")
        model.save_pretrained(output / "adapter", safe_serialization=True)
        tokenizer.save_pretrained(output / "tokenizer")
        torch.set_num_threads(config["evaluation_threads"])
        after_validation = loss_evaluation(model, encoded["validation"], tokenizer.pad_token_id, config["loss_batch_size"])
        after_loss = loss_evaluation(model, encoded["test"], tokenizer.pad_token_id, config["loss_batch_size"])
        after_generation, raw_after = generation_evaluation(model, tokenizer, splits["test"], config["generation_batch_size"], config["generation_tokens"])
        write_json(output / "predictions-after-local.json", raw_after)
        assert_cpu_model(model)
        metric = {"experiment": "CPU-only generative answer extraction from complete SQuAD passages", "completed_utc": datetime.now(timezone.utc).isoformat(),
                  "model": {"id": config["model_id"], "revision": config["revision"], "resolved_revision": getattr(base.config, "_commit_hash", None),
                            "base_parameters": base_parameters, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad)},
                  "dataset": provenance, "before": before, "after": {"validation": after_validation, "test_loss": after_loss, "test_generation": after_generation},
                  "selection": {"criterion": "minimum validation response-token loss, including initial model", "step": best_step, "test_used_for_selection": False},
                  "training": {"optimizer_steps": config["optimizer_steps"], "examples_seen": config["optimizer_steps"] * config["batch_size"],
                               "unique_examples_seen": len(seen), "target_tokens": target_tokens, "input_tokens": input_tokens,
                               "compute_seconds": compute, "loop_seconds_including_validation": loop_seconds,
                               "selected_adapter_l2_change": math.sqrt(sum(float((best_adapter[k] - v).square().sum()) for k, v in initial.items()))},
                  "optimizations": {"training_threads": config["training_threads"], "evaluation_threads": config["evaluation_threads"],
                                    "response_only_vocabulary_projection": True, "loss_length_sorting": True,
                                    "loss_batch_size": config["loss_batch_size"], "generation_batch_size": config["generation_batch_size"],
                                    "generation": "left-padded greedy batches, length sorted, restored source order; per-row EOS trimming"},
                  "cpu_evidence": {"torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                                   "parameter_devices": sorted({str(p.device) for p in model.parameters()}),
                                   "optimizer_tensor_devices": sorted({str(v.device) for state in optimizer.state.values() for v in state.values() if isinstance(v, torch.Tensor)}), "dtype": "float32"},
                  "environment": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "psutil")},
                  "elapsed_seconds": time.perf_counter() - started, "sampled_peak_process_rss_bytes": memory.peak_bytes,
                  "resolved_config_sha256": sha256((output / "resolved_config.json").read_bytes()),
                  "training_curve_sha256": sha256((output / "training_curve.json").read_bytes()),
                  "data_provenance_sha256": sha256((output / "data_provenance.json").read_bytes()),
                  "adapter_sha256": directory_hashes(output / "adapter"),
                  "tokenizer_sha256": directory_hashes(output / "tokenizer"),
                  "local_prediction_sha256": {p.name: sha256(p.read_bytes()) for p in output.glob("predictions-*-local.json")},
                  "limitations": ["This is a filtered short-passage/short-answer subset, not the full SQuAD benchmark or its hidden test set.",
                                  "The historical Wikipedia/SQuAD material may have appeared in the base model's pretraining.",
                                  "Every question is answerable; abstention, unanswerable questions, long documents and OCR are not evaluated.",
                                  "Article and exact-passage separation cannot exclude all related facts or near-duplicate material.",
                                  "Loss selects checkpoints; improved loss does not guarantee improved generated exact match or F1.",
                                  "Batched generation may have small floating-point differences from serial generation; max token caps can truncate answers.",
                                  "Only IDs, hashes and scores are published by default; raw passages, references and generated answers remain in local artifacts.",
                                  "No end-to-end speedup claim is made without a matched document-task baseline run; transferred optimizations are disclosed."]}
        write_json(output / "metrics.json", metric)
        write_json(output / "run_status.json", {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"stage": "complete", "before_exact_match": before_generation["exact_match"], "after_exact_match": after_generation["exact_match"],
                          "before_f1": before_generation["token_f1"], "after_f1": after_generation["token_f1"], "training_seconds": loop_seconds}), flush=True)
        return metric


def load_inference(directory: Path, threads: int):
    if json.loads((directory / "run_status.json").read_text())["status"] != "complete":
        raise ValueError("Load only completed document runs.")
    config = json.loads((directory / "resolved_config.json").read_text())
    metrics = json.loads((directory / "metrics.json").read_text())
    verify_saved_artifacts(directory, metrics)
    if config["model_id"] != metrics["model"]["id"] or config["revision"] != metrics["model"]["revision"]:
        raise ValueError("Saved model revision differs from the run.")
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("CPU-only PyTorch is required.")
    torch.set_num_threads(threads)
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True, trust_remote_code=False)
    base = AutoModelForCausalLM.from_pretrained(config["model_id"], revision=config["revision"], local_files_only=True,
                                               trust_remote_code=False, torch_dtype=torch.float32, attn_implementation="sdpa").cpu()
    model = PeftModel.from_pretrained(base, directory / "adapter", is_trainable=False).eval()
    model.generation_config.do_sample = False
    for key in ("temperature", "top_k", "top_p"):
        setattr(model.generation_config, key, None)
    assert_cpu_model(model)
    return model, tokenizer, config, metrics


@torch.inference_mode()
def generate_answer(model, tokenizer, config: dict, context: str, question: str) -> str:
    text = render_prompt(tokenizer, {"prompt": prompt_text(context, question)})
    inputs = tokenizer(text, add_special_tokens=False, return_tensors="pt")
    if inputs["input_ids"].numel() > config["max_prompt_tokens"]:
        raise ValueError("This complete passage/question exceeds the trained prompt budget; no automatic truncation.")
    generated = model.generate(**inputs, max_new_tokens=config["generation_tokens"], do_sample=False,
                               eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id, use_cache=True)
    ids = trim_generated(generated[0, inputs["input_ids"].shape[1]:].tolist(), tokenizer.eos_token_id)
    return tokenizer.decode(ids, skip_special_tokens=True)


def predict(directory: Path, context: str, question: str, threads: int) -> dict:
    model, tokenizer, config, _ = load_inference(directory, threads)
    answer = generate_answer(model, tokenizer, config, context, question)
    return {"answer": answer, "verbatim_span_in_passage": bool(answer.strip()) and answer.strip().casefold() in context.casefold(), "device": "cpu"}


def verify_reload(directory: Path, prepared: Path, threads: int, limit: int = 3) -> dict:
    """Reload once, then compare the first fixed test IDs with saved batched output.

    This is a replay smoke check, never a checkpoint or hyperparameter selector.
    Serial/batched floating-point differences are reported instead of hidden.
    """
    if limit < 1:
        raise ValueError("Reload verification requires at least one example.")
    model, tokenizer, config, metrics = load_inference(directory, threads)
    splits, provenance = load_prepared(prepared, config)
    if provenance["split_manifest_sha256"] != metrics["dataset"]["split_manifest_sha256"]:
        raise ValueError("Replay data differs from the saved experiment.")
    expected = {record["id"]: record for record in metrics["after"]["test_generation"]["records"]}
    records = []
    for source in splits["test"][:limit]:
        answer = generate_answer(model, tokenizer, config, source["context"], source["question"])
        digest = sha256(answer.encode("utf-8"))
        records.append({"id": source["id"], "prediction_sha256": digest,
                        "expected_prediction_sha256": expected[source["id"]]["prediction_sha256"],
                        "matches_saved_batched_output": digest == expected[source["id"]]["prediction_sha256"]})
    result = {"examples": len(records), "matched": sum(record["matches_saved_batched_output"] for record in records),
              "selection": "first test IDs in immutable split order, no selection by output quality",
              "reloaded_from_saved_artifacts": True, "all_artifact_checksums_verified": True,
              "mode": "serial greedy reload compared with original length-sorted batched generation",
              "parameter_devices": sorted({str(p.device) for p in model.parameters()}), "records": records}
    write_json(directory / "reload_verification.json", result)
    return result


def export_public(directory: Path, destination: Path, prefix: str = "document") -> dict:
    """Byte-preserving publication of an explicit raw-text-free artifact whitelist."""
    if not re.fullmatch(r"[A-Za-z0-9_-]+", prefix):
        raise ValueError("Public export prefix must contain only letters, numbers, underscores or hyphens.")
    if json.loads((directory / "run_status.json").read_text())["status"] != "complete":
        raise ValueError("Export only completed document runs.")
    metrics = json.loads((directory / "metrics.json").read_text())
    verify_saved_artifacts(directory, metrics)
    names = {"metrics.json": "metrics", "resolved_config.json": "config", "training_curve.json": "training-curve",
             "data_provenance.json": "data-provenance", "split_manifest.json": "split-manifest"}
    files = [(directory / source, destination / f"{prefix}-{suffix}.json") for source, suffix in names.items()]
    for source, target in files:
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise FileExistsError(f"Public artifact already exists with different bytes: {target}")
    destination.mkdir(parents=True, exist_ok=True)
    hashes = {}
    for source, target in files:
        payload = source.read_bytes()
        target.write_bytes(payload)
        if target.read_bytes() != payload:
            raise RuntimeError("Public artifact byte verification failed.")
        hashes[target.name] = sha256(payload)
    return {"files_sha256": hashes, "raw_passages_questions_answers_or_predictions_exported": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/document-task.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path, default=Path(".cache/document-data"))
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--prepared-data", type=Path)
    parser.add_argument("--training-threads", type=int)
    parser.add_argument("--evaluation-threads", type=int)
    parser.add_argument("--loss-batch-size", type=int)
    parser.add_argument("--generation-batch-size", type=int)
    parser.add_argument("--steps", type=int)
    parser.add_argument("--load", type=Path)
    parser.add_argument("--context")
    parser.add_argument("--question")
    parser.add_argument("--verify-reload", type=Path)
    parser.add_argument("--reload-limit", type=int, default=3)
    parser.add_argument("--export-run", type=Path)
    parser.add_argument("--results-directory", type=Path, default=Path("results"))
    parser.add_argument("--export-prefix", default="document")
    args = parser.parse_args()
    if args.export_run:
        print(json.dumps(export_public(args.export_run, args.results_directory, args.export_prefix)))
        return
    if args.verify_reload:
        if args.prepared_data is None:
            parser.error("--verify-reload requires --prepared-data")
        print(json.dumps(verify_reload(args.verify_reload, args.prepared_data, args.evaluation_threads or 4, args.reload_limit)))
        return
    if args.load:
        if not args.context or not args.question:
            parser.error("--load requires --context and --question")
        print(json.dumps(predict(args.load, args.context, args.question, args.evaluation_threads or 4)))
        return
    if args.output is None:
        parser.error("--output is required")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Use a new output directory; previous results are preserved.")
    config = json.loads(args.config.read_text())
    for key in ("training_threads", "evaluation_threads", "loss_batch_size", "generation_batch_size"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    if args.steps is not None:
        config["optimizer_steps"] = args.steps
    for key in ("training_threads", "evaluation_threads", "loss_batch_size", "generation_batch_size", "batch_size", "optimizer_steps", "validation_every"):
        if type(config[key]) is not int or config[key] < 1:
            raise ValueError(f"Invalid positive integer: {key}")
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / "resolved_config.json", config)
    write_json(args.output / "run_status.json", {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()})
    try:
        if args.prepare_only:
            splits, provenance = prepare(config, args.output, args.cache, args.download_data)
            write_json(args.output / "run_status.json", {"status": "prepared"})
            print(json.dumps({"counts": provenance["counts"], "eligibility": provenance["eligibility"], "selection": provenance["selection"]}), flush=True)
        else:
            _train(config, args.output, args.cache, args.download_data, args.prepared_data)
    except BaseException as error:
        write_json(args.output / "run_status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error_type": type(error).__name__, "error": str(error)[:500]})
        raise


if __name__ == "__main__":
    main()
