"""CPU LoRA classification on the public UCI SMS Spam Collection.

This is constrained two-label classification, not free-form generation. The
causal language model predicts one of the single-token labels 0 (ham) / 1 (spam).
Only those two next-token logits enter the loss and decision. Raw SMS messages
stay in the ignored local run directory; numerical reports contain hashes only.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import random
import re
import time
import unicodedata
import urllib.request
import zipfile

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from peft import LoraConfig, PeftModel, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from transformers import AutoModelForCausalLM, AutoTokenizer

from .training import PeakRSS, adapter_snapshot, assert_cpu_model, write_json

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
DATA_SHA256 = "1587ea43e58e82b14ff1f5425c88e17f8496bfcdb67a583dbff9eefaf9963ce3"
SOURCE_PAGE = "https://archive.ics.uci.edu/dataset/228/sms+spam+collection"
LABELS = {"ham": 0, "spam": 1}
INSTRUCTION = "Classify the SMS message. Reply with 0 for legitimate personal messages (ham), or 1 for unsolicited advertising, scams, or spam.\nSMS: "


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def load_sms(archive: Path, download: bool = False) -> tuple[list[dict], dict]:
    if not archive.exists():
        if not download:
            raise FileNotFoundError(f"Dataset is not cached at {archive}; pass --download-data.")
        archive.parent.mkdir(parents=True, exist_ok=True)
        with urllib.request.urlopen(DATA_URL, timeout=60) as response:
            raw = response.read(2_000_001)
        if len(raw) > 2_000_000 or sha256(raw) != DATA_SHA256:
            raise ValueError("Dataset archive failed its pinned SHA256/size check.")
        archive.write_bytes(raw)
    raw = archive.read_bytes()
    if sha256(raw) != DATA_SHA256:
        raise ValueError("Dataset archive does not match the pinned SHA256.")
    with zipfile.ZipFile(io.BytesIO(raw)) as zipped:
        source = zipped.read("SMSSpamCollection")
    rows = []
    for index, line in enumerate(source.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        label, text = line.split("\t", 1)
        if label not in LABELS or not text.strip():
            raise ValueError(f"Invalid source row {index}.")
        rows.append({"id": index, "text": text, "label": LABELS[label], "text_sha256": sha256(text.encode("utf-8"))})
    return rows, {
        "name": "SMS Spam Collection", "source_url": DATA_URL, "source_page": SOURCE_PAGE,
        "citation": "Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection. UCI Machine Learning Repository. DOI:10.24432/C5CC84",
        "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "archive_sha256": DATA_SHA256, "source_file_sha256": sha256(source), "source_rows": len(rows),
    }


def normalize_template(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"(?:https?://|www\.)\S+", " url ", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", " email ", text)
    text = re.sub(r"\d+", " number ", text)
    return " ".join(re.findall(r"[a-z]+", text))


def deduplicate(rows: list[dict], similarity: float = 0.80) -> tuple[list[dict], dict]:
    """Keep one example per connected near-duplicate group, before any split.

    Exact templates ignore case, punctuation, digits, URLs and email addresses.
    For templates of >=10 words, additionally union word-trigram Jaccard >=0.8.
    This is a documented heuristic; unseen sender/source overlap can remain.
    """
    if not 0 < similarity <= 1:
        raise ValueError("Near-duplicate similarity must be in (0,1].")
    parent = list(range(len(rows)))

    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(a, b):
        a, b = find(a), find(b)
        if a != b:
            parent[max(a, b)] = min(a, b)

    exact, index = {}, defaultdict(list)
    shingles = []
    for i, row in enumerate(rows):
        template = normalize_template(row["text"])
        if template in exact:
            union(i, exact[template])
        else:
            exact[template] = i
        words = template.split()
        grams = {tuple(words[j:j + 3]) for j in range(len(words) - 2)} if len(words) >= 10 else set()
        candidates = set()
        for gram in grams:
            candidates.update(index[gram])
        for j in candidates:
            other = shingles[j]
            if min(len(grams), len(other)) / max(len(grams), len(other)) < similarity:
                continue
            if len(grams & other) / len(grams | other) >= similarity:
                union(i, j)
        shingles.append(grams)
        for gram in grams:
            index[gram].append(i)
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(rows[i])
    kept, conflicts = [], 0
    for members in groups.values():
        if len({row["label"] for row in members}) != 1:
            conflicts += len(members)
            continue
        row = dict(members[0])
        row["duplicate_group_size"] = len(members)
        row["group_sha256"] = sha256(
            ",".join(str(member["id"]) for member in members).encode("utf-8")
        )
        kept.append(row)
    return kept, {
        "original_rows": len(rows), "retained_representatives": len(kept),
        "duplicate_or_conflicting_rows_removed": len(rows) - len(kept),
        "conflicting_label_rows_removed": conflicts, "near_duplicate_trigram_jaccard_threshold": similarity,
        "method": "NFKC/lowercase/punctuation/digit/URL/email normalized exact templates; >=10-word templates also grouped by word-trigram Jaccard, transitive connected components; conflicting groups removed; one representative retained before splitting.",
    }


def make_splits(rows: list[dict], train_size: int, validation_size: int, test_size: int, seed: int) -> dict[str, list[dict]]:
    sizes = {"train": train_size, "validation": validation_size, "test": test_size}
    if min(sizes.values()) < 2 or sum(sizes.values()) > len(rows):
        raise ValueError("Split sizes must be >=2 and fit the deduplicated dataset.")
    grouped = {label: [row for row in rows if row["label"] == label] for label in (0, 1)}
    if not all(grouped.values()):
        raise ValueError("Both labels are needed.")
    rng = random.Random(seed)
    for values in grouped.values():
        rng.shuffle(values)
    positions = {0: 0, 1: 0}
    splits = {}
    rate = len(grouped[1]) / len(rows)
    # Test/validation are allocated first so changing train size leaves them fixed.
    for name in ("test", "validation", "train"):
        positive = max(1, min(sizes[name] - 1, round(sizes[name] * rate)))
        chosen = []
        for label, count in ((0, sizes[name] - positive), (1, positive)):
            start = positions[label]
            chosen.extend(grouped[label][start:start + count])
            positions[label] += count
        if len(chosen) != sizes[name]:
            raise ValueError("Insufficient rows of a label for the stratified split.")
        rng.shuffle(chosen)
        splits[name] = chosen
    all_ids = [row["id"] for split in splits.values() for row in split]
    if len(all_ids) != len(set(all_ids)):
        raise RuntimeError("Overlapping split IDs.")
    return splits


def classification_metrics(expected: list[int], predicted: list[int]) -> dict:
    if not expected or len(expected) != len(predicted) or set(expected + predicted) - {0, 1}:
        raise ValueError("Expected matching, nonempty binary labels.")
    matrix = [[0, 0], [0, 0]]
    for truth, pred in zip(expected, predicted):
        matrix[truth][pred] += 1
    per_class = {}
    for label, name in ((0, "ham"), (1, "spam")):
        tp = matrix[label][label]
        fp = matrix[1 - label][label]
        fn = matrix[label][1 - label]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0
        per_class[name] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}
    return {"examples": len(expected), "correct": matrix[0][0] + matrix[1][1],
            "accuracy": (matrix[0][0] + matrix[1][1]) / len(expected),
            "macro_f1": sum(value["f1"] for value in per_class.values()) / 2,
            "confusion_matrix_true_rows_predicted_columns_ham_spam": matrix, "per_class": per_class}


def nb_tokens(text: str) -> list[str]:
    # Normalize URLs/numbers consistently but preserve repeated words for counts.
    return normalize_template(text).split()


def naive_bayes(splits: dict[str, list[dict]], alphas=(0.1, 0.5, 1.0, 2.0), recipe_output: Path | None = None) -> dict:
    started = time.perf_counter()
    counts = {label: Counter() for label in (0, 1)}
    documents = Counter(row["label"] for row in splits["train"])
    for row in splits["train"]:
        counts[row["label"]].update(nb_tokens(row["text"]))
    vocabulary = set(counts[0]) | set(counts[1])
    totals = {label: sum(counts[label].values()) for label in (0, 1)}

    def predict(rows, alpha):
        if alpha <= 0:
            raise ValueError("Naive Bayes smoothing must be positive.")
        predictions = []
        for row in rows:
            words = Counter(word for word in nb_tokens(row["text"]) if word in vocabulary)
            scores = []
            for label in (0, 1):
                denominator = totals[label] + alpha * len(vocabulary)
                value = math.log(documents[label] / len(splits["train"]))
                value += sum(number * math.log((counts[label][word] + alpha) / denominator) for word, number in words.items())
                scores.append(value)
            predictions.append(int(scores[1] > scores[0]))
        return predictions

    candidates = []
    for alpha in alphas:
        metric = classification_metrics([row["label"] for row in splits["validation"]], predict(splits["validation"], alpha))
        candidates.append({"alpha": alpha, **metric})
    best = max(candidates, key=lambda item: item["macro_f1"])
    if recipe_output is not None:
        write_json(recipe_output, {"alpha": best["alpha"], "class_document_counts": dict(documents),
                                 "word_counts": {label: dict(values) for label, values in counts.items()},
                                 "normalization": "normalize_template v1", "vocabulary_size": len(vocabulary)})
    train_seconds = time.perf_counter() - started
    test_started = time.perf_counter()
    predictions = predict(splits["test"], best["alpha"])
    test_seconds = time.perf_counter() - test_started
    metric = classification_metrics([row["label"] for row in splits["test"]], predictions)
    return {"method": "Multinomial Naive Bayes, normalized word counts, training-only vocabulary and class priors; unseen vocabulary ignored",
            "selected_alpha": best["alpha"], "selection": "highest validation macro F1; ties keep first declared alpha",
            "validation_candidates": candidates, "test": metric, "training_and_selection_seconds": train_seconds,
            "test_seconds": test_seconds, "vocabulary_size": len(vocabulary),
            "test_predictions": [{"id": row["id"], "text_sha256": row["text_sha256"], "expected": row["label"], "predicted": pred} for row, pred in zip(splits["test"], predictions)]}


def encode_message(tokenizer, text: str, max_length: int) -> tuple[list[int], bool]:
    def render(content):
        return tokenizer.apply_chat_template([{"role": "user", "content": content}], tokenize=False, add_generation_prompt=True)

    def encode(content):
        return tokenizer.encode(render(INSTRUCTION + content), add_special_tokens=False)

    ids = encode(text)
    if len(ids) <= max_length:
        return ids, False
    # Explicitly truncate only message content, preserving instruction and chat suffix.
    message_ids = tokenizer.encode(text, add_special_tokens=False)
    budget = max_length - len(encode(""))
    if budget < 1:
        raise ValueError("max_sequence_length cannot fit the instruction/chat template.")
    for keep in range(min(budget, len(message_ids)), -1, -1):
        ids = encode(tokenizer.decode(message_ids[:keep], skip_special_tokens=False))
        if len(ids) <= max_length:
            return ids, True
    raise RuntimeError("Could not fit the classification prompt.")


def encode_split(tokenizer, rows, max_length):
    encoded = []
    for row in rows:
        ids, truncated = encode_message(tokenizer, row["text"], max_length)
        encoded.append({"input_ids": ids, "label": row["label"], "id": row["id"],
                        "text_sha256": row["text_sha256"], "truncated": truncated})
    return encoded


def collate_classification(rows: list[dict], pad_token_id: int) -> tuple[dict, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in rows)
    ids, masks = [], []
    for row in rows:
        count = width - len(row["input_ids"])
        ids.append([pad_token_id] * count + row["input_ids"])
        masks.append([0] * count + [1] * len(row["input_ids"]))
    mask = torch.tensor(masks, dtype=torch.long, device="cpu")
    return {"input_ids": torch.tensor(ids, dtype=torch.long, device="cpu"), "attention_mask": mask,
            "position_ids": (mask.cumsum(-1) - 1).clamp(min=0)}, torch.tensor([row["label"] for row in rows], dtype=torch.long, device="cpu")


def label_logits(model, batch, token_ids):
    # logits_to_keep avoids a vocabulary projection at every prompt position.
    return model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1, token_ids]


@torch.inference_mode()
def evaluate(model, encoded, pad_token_id, token_ids, batch_size):
    model.eval()
    started = time.perf_counter()
    records, losses = [], []
    for offset in range(0, len(encoded), batch_size):
        rows = encoded[offset:offset + batch_size]
        batch, labels = collate_classification(rows, pad_token_id)
        logits = label_logits(model, batch, token_ids)
        nll = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
        probabilities = logits.softmax(-1)[:, 1]
        predictions = logits.argmax(-1)
        for row, truth, pred, probability, loss in zip(rows, labels.tolist(), predictions.tolist(), probabilities.tolist(), nll.tolist()):
            records.append({"id": row["id"], "text_sha256": row["text_sha256"], "expected": truth, "predicted": pred,
                            "spam_probability_conditioned_on_two_labels": probability, "binary_label_nll": loss,
                            "input_tokens": len(row["input_ids"]), "message_truncated": row["truncated"]})
            losses.append(loss)
    return {**classification_metrics([row["expected"] for row in records], [row["predicted"] for row in records]),
            "binary_label_loss": sum(losses) / len(losses), "seconds": time.perf_counter() - started, "records": records}


def compact(metric):
    return {key: value for key, value in metric.items() if key != "records"}


def train(config: dict, output: Path, archive: Path, download: bool = False):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Refusing to overwrite a nonempty experiment directory.")
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "resolved_config.json", config)
    write_json(output / "run_status.json", {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()})
    try:
        return _train(config, output, archive, download)
    except BaseException as error:
        write_json(output / "run_status.json", {"status": "interrupted" if isinstance(error, KeyboardInterrupt) else "failed", "error_type": type(error).__name__, "error": str(error)[:500]})
        raise


def _train(config, output, archive, download):
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("Requires a CPU-only PyTorch build.")
    for name in ("threads", "optimizer_steps", "batch_size", "evaluation_batch_size", "validation_every"):
        if config[name] < 1:
            raise ValueError(f"{name} must be positive.")
    if config["batch_size"] % 2:
        raise ValueError("Balanced training sampler requires an even batch size.")
    execution = config.get("execution")
    training_threads = evaluation_threads = config["threads"]
    if execution is not None:
        from .pipeline_speed import optimized_evaluate, optimized_train_step

        training_execution = execution["training"]
        evaluation_execution = execution["evaluation"]
        for count in (training_execution["threads"], evaluation_execution["threads"], evaluation_execution["batch_size"]):
            if type(count) is not int or not 1 <= count <= 256:
                raise ValueError("Execution thread and evaluation batch counts must be integers from 1 to 256.")
        microbatch_size = training_execution["microbatch_size"]
        if type(microbatch_size) is not int or not 1 <= microbatch_size <= config["batch_size"]:
            raise ValueError("Execution microbatch size must be a positive integer no larger than the global batch.")
        if type(evaluation_execution["sort_by_length"]) is not bool:
            raise ValueError("Execution sort_by_length must be a boolean.")
        training_threads = training_execution["threads"]
        evaluation_threads = evaluation_execution["threads"]
    training_threads_observed, evaluation_threads_observed = set(), set()
    torch.set_num_threads(training_threads)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    started = time.perf_counter()
    with PeakRSS() as memory:
        rows, provenance = load_sms(archive, download)
        rows, dedup = deduplicate(rows, config["near_duplicate_similarity"])
        splits = make_splits(rows, config["train_size"], config["validation_size"], config["test_size"], config["seed"])
        local_data = output / "data"
        local_data.mkdir()
        manifests = {}
        for name, values in splits.items():
            payload = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in values)
            (local_data / (name + ".jsonl")).write_text(payload, encoding="utf-8")
            manifests[name] = [{key: value for key, value in row.items() if key != "text"} for row in values]
        write_json(output / "split_manifest.json", manifests)
        nb = naive_bayes(splits, recipe_output=output / "naive_bayes.json")
        majority = Counter(row["label"] for row in splits["train"]).most_common(1)[0][0]
        majority_metric = classification_metrics([row["label"] for row in splits["test"]], [majority] * len(splits["test"]))
        write_json(output / "classical_baselines.json", {"majority": majority_metric, "naive_bayes": nb})
        print(json.dumps({"stage": "loading_model", "model": config["model_id"], "deduplicated": dedup, "naive_bayes_test_macro_f1": nb["test"]["macro_f1"]}), flush=True)
        options = {"revision": config["revision"], "local_files_only": config.get("local_files_only", True), "trust_remote_code": False}
        tokenizer = AutoTokenizer.from_pretrained(config["model_id"], **options)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokens = [tokenizer.encode(label, add_special_tokens=False) for label in ("0", "1")]
        if any(len(ids) != 1 for ids in tokens) or tokens[0] == tokens[1]:
            raise ValueError("Both labels must be distinct single tokens for this classifier.")
        token_ids = [ids[0] for ids in tokens]
        base = AutoModelForCausalLM.from_pretrained(config["model_id"], torch_dtype=torch.float32, attn_implementation="sdpa", **options).to("cpu")
        base_parameters = sum(parameter.numel() for parameter in base.parameters())
        model = get_peft_model(base, LoraConfig(task_type="CAUSAL_LM", r=config["lora_rank"], lora_alpha=config["lora_alpha"], lora_dropout=0.0, target_modules=config["lora_target_modules"], bias="none"))
        assert_cpu_model(model)
        initial = adapter_snapshot(model)
        encoded = {name: encode_split(tokenizer, values, config["max_sequence_length"]) for name, values in splits.items()}
        truncated = {name: sum(row["truncated"] for row in values) for name, values in encoded.items()}

        def evaluate_for_run(values):
            if execution is None:
                result = evaluate(model, values, tokenizer.pad_token_id, token_ids, config["evaluation_batch_size"])
            else:
                result = optimized_evaluate(model, values, tokenizer.pad_token_id, token_ids,
                                            batch_size=evaluation_execution["batch_size"],
                                            sort_by_length=evaluation_execution["sort_by_length"],
                                            threads=evaluation_threads)
            evaluation_threads_observed.add(torch.get_num_threads())
            return result

        print(json.dumps({"stage": "baseline_evaluation", "truncated_message_counts": truncated}), flush=True)
        before_validation = evaluate_for_run(encoded["validation"])
        before_test = evaluate_for_run(encoded["test"])
        write_json(output / "baseline.json", {"validation": before_validation, "test": before_test})
        best = (before_validation["macro_f1"], -before_validation["binary_label_loss"])
        best_step, best_adapter, best_seen = 0, initial, set()
        optimizer = torch.optim.AdamW([parameter for parameter in model.parameters() if parameter.requires_grad], lr=config["learning_rate"], weight_decay=0.0)
        rng = random.Random(config["seed"])
        indices = {label: [i for i, row in enumerate(encoded["train"]) if row["label"] == label] for label in (0, 1)}
        for values in indices.values():
            rng.shuffle(values)
        positions = {0: 0, 1: 0}
        curve, seen = [], set()
        schedule_ids = []
        compute_seconds, input_tokens = 0.0, 0
        training_started = time.perf_counter()
        for step in range(1, config["optimizer_steps"] + 1):
            selected = []
            for label in (0, 1):
                for _ in range(config["batch_size"] // 2):
                    if positions[label] == len(indices[label]):
                        rng.shuffle(indices[label])
                        positions[label] = 0
                    selected.append(indices[label][positions[label]])
                    positions[label] += 1
            rng.shuffle(selected)
            schedule_ids.append([encoded["train"][index]["id"] for index in selected])
            seen.update(selected)
            if execution is None:
                batch, targets = collate_classification([encoded["train"][i] for i in selected], tokenizer.pad_token_id)
                model.train()
                optimizer.zero_grad(set_to_none=True)
                step_started = time.perf_counter()
                logits = label_logits(model, batch, token_ids)
                loss = torch.nn.functional.cross_entropy(logits, targets)
                if not torch.isfinite(loss):
                    raise RuntimeError("Nonfinite training loss.")
                loss.backward()
                torch.nn.utils.clip_grad_norm_([parameter for parameter in model.parameters() if parameter.requires_grad], 1.0, error_if_nonfinite=True)
                optimizer.step()
                elapsed = time.perf_counter() - step_started
                compute_seconds += elapsed
                input_tokens += int(batch["attention_mask"].sum())
                record = {"step": step, "binary_label_loss": float(loss.detach()), "seconds": elapsed}
            else:
                step_result = optimized_train_step(model, optimizer, [encoded["train"][i] for i in selected],
                                                   tokenizer.pad_token_id, token_ids,
                                                   microbatch_size=training_execution["microbatch_size"],
                                                   threads=training_threads)
                compute_seconds += step_result["seconds"]
                input_tokens += step_result["input_tokens"]
                record = {"step": step, "binary_label_loss": step_result["binary_label_loss"], "seconds": step_result["seconds"],
                          "execution": {key: value for key, value in step_result.items() if key not in ("binary_label_loss", "seconds")}}
            training_threads_observed.add(torch.get_num_threads())
            if step % config["validation_every"] == 0 or step == config["optimizer_steps"]:
                validation = evaluate_for_run(encoded["validation"])
                record["validation"] = compact(validation)
                candidate = (validation["macro_f1"], -validation["binary_label_loss"])
                if candidate > best:
                    best, best_step, best_adapter = candidate, step, adapter_snapshot(model)
                    best_seen = set(seen)
            curve.append(record)
            write_json(output / "training_curve.json", curve)
            if step % 5 == 0 or step == 1:
                print(json.dumps({"stage": "training", **record}), flush=True)
        training_seconds = time.perf_counter() - training_started
        set_peft_model_state_dict(model, best_adapter)
        restored = get_peft_model_state_dict(model)
        if any(not torch.equal(restored[name].cpu(), tensor) for name, tensor in best_adapter.items()):
            raise RuntimeError("Selected checkpoint restoration failed.")
        assert_cpu_model(model)
        model.save_pretrained(output / "adapter", safe_serialization=True)
        tokenizer.save_pretrained(output / "tokenizer")
        task_recipe = {**config, "label_token_ids": token_ids, "label_map": LABELS, "instruction": INSTRUCTION,
                       "scoring": "argmax of two next-token label logits; probabilities normalized over labels only", "selected_step": best_step}
        write_json(output / "classifier.json", task_recipe)
        selected_rows = [splits["train"][index] for index in sorted(best_seen)]
        selected_manifest = [{key: value for key, value in row.items() if key != "text"} for row in selected_rows]
        write_json(output / "selected_training_manifest.json", selected_manifest)
        matched_nb = naive_bayes({**splits, "train": selected_rows}, recipe_output=output / "naive_bayes_matched.json") if selected_rows else None
        print(json.dumps({"stage": "final_evaluation", "selected_step": best_step}), flush=True)
        after_validation = evaluate_for_run(encoded["validation"])
        after_test = evaluate_for_run(encoded["test"])
        delta = math.sqrt(sum(float((best_adapter[name] - tensor).square().sum()) for name, tensor in initial.items()))
        metric = {
            "experiment": "CPU-only constrained SMS spam classification with LoRA",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "model": {"id": config["model_id"], "revision": config["revision"], "resolved_revision": getattr(base.config, "_commit_hash", None), "base_parameters": base_parameters,
                      "trainable_parameters": sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)},
            "dataset": {**provenance, "deduplication": dedup, "split_seed": config["seed"],
                        "counts": {name: len(values) for name, values in splits.items()},
                        "class_counts": {name: dict(Counter(str(row["label"]) for row in values)) for name, values in splits.items()},
                        "split_manifest_sha256": sha256((output / "split_manifest.json").read_bytes()),
                        "message_truncation_counts": truncated, "truncation_policy": "Only long message content is truncated from the right to fit the configured prompt length; chat template and instruction remain complete."},
            "classifier": {"label_map": LABELS, "label_token_ids": token_ids, "scoring": task_recipe["scoring"], "training_objective": "binary cross entropy through the two label logits, balanced sampler with equal class counts per training batch"},
            "selection": {"criterion": "highest validation macro F1, then lowest validation binary-label loss; baseline included", "selected_step": best_step, "test_used_for_selection": False},
            "before": {"validation": before_validation, "test": before_test},
            "after": {"validation": after_validation, "test": after_test},
            "classical_baselines": {"majority_label_from_training": majority, "majority_test": majority_metric, "naive_bayes": nb,
                                    "naive_bayes_matched_selected_training_rows": matched_nb,
                                    "matched_baseline_policy": "Train on the unique rows seen up to the validation-selected LoRA checkpoint; select smoothing on the same validation set. Null if the untrained checkpoint is selected."},
            "training": {"optimizer_steps": config["optimizer_steps"], "examples_seen_with_repetition": config["optimizer_steps"] * config["batch_size"], "unique_training_examples_seen": len(seen),
                         "selected_checkpoint_examples_seen_with_repetition": best_step * config["batch_size"],
                         "selected_checkpoint_unique_training_examples_seen": len(best_seen),
                         "selected_training_manifest_sha256": sha256((output / "selected_training_manifest.json").read_bytes()), "input_tokens": input_tokens,
                         "sample_schedule_sha256": sha256(json.dumps(schedule_ids, separators=(",", ":")).encode()),
                         "compute_seconds_excluding_validation": compute_seconds, "wall_seconds_including_validation": training_seconds,
                         "compute_timing_includes_collation": execution is not None,
                         "input_tokens_per_compute_second": input_tokens / compute_seconds, "selected_adapter_l2_change": delta},
            "cpu_evidence": {"torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "parameter_devices": sorted({str(p.device) for p in model.parameters()}), "buffer_devices": sorted({str(b.device) for b in model.buffers()}), "threads": torch.get_num_threads(), "dtype": "float32",
                             "training_threads_configured": training_threads, "evaluation_threads_configured": evaluation_threads,
                             "training_threads_observed": sorted(training_threads_observed), "evaluation_threads_observed": sorted(evaluation_threads_observed)},
            "environment": {"packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "psutil")}},
            "sampled_peak_process_rss_bytes": memory.peak_bytes, "elapsed_seconds": time.perf_counter() - started,
            "adapter_sha256": {path.name: sha256(path.read_bytes()) for path in (output / "adapter").glob("*") if path.is_file()},
            "naive_bayes_recipe_sha256": sha256((output / "naive_bayes.json").read_bytes()),
            "limitations": ["Historical public SMS dataset, not a current production spam filter.", "Pretrained-model contamination cannot be ruled out; the dataset predates the base model.", "Template and near-duplicate grouping is heuristic, not sender/source disjointness.", "This is discriminative two-label classification, not unrestricted generation accuracy.", "Training uses a balanced sampler; reported probabilities are not calibrated deployment spam probabilities.", "Classical baseline sees all training rows, while LoRA sees the recorded fixed-budget sampled rows.", "No guarantee these findings generalize to other tasks or language-model capabilities.", "Sampled process RSS is not exact peak or total system memory."]}
        write_json(output / "metrics.json", metric)
        write_json(output / "run_status.json", {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat()})
        print(json.dumps({"stage": "complete", "before_accuracy": before_test["accuracy"], "after_accuracy": after_test["accuracy"], "before_macro_f1": before_test["macro_f1"], "after_macro_f1": after_test["macro_f1"], "naive_bayes_macro_f1": nb["test"]["macro_f1"], "training_seconds": training_seconds}), flush=True)
        return metric


def predict_local(directory: Path, text: str, threads: int = 4):
    status = json.loads((directory / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError("Only a completed training run can be loaded.")
    config = json.loads((directory / "classifier.json").read_text(encoding="utf-8"))
    resolved = json.loads((directory / "resolved_config.json").read_text(encoding="utf-8"))
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if any(config.get(name) != value for name, value in resolved.items()):
        raise ValueError("Classifier recipe does not match the resolved training configuration.")
    if config.get("instruction") != INSTRUCTION or config.get("label_map") != LABELS or not re.fullmatch(r"[0-9a-f]{40}", config["revision"]):
        raise ValueError("Classifier prompt, labels or pinned revision are invalid.")
    if config["model_id"] != metrics["model"]["id"] or config["revision"] != metrics["model"]["revision"]:
        raise ValueError("Classifier recipe does not match the recorded model/revision.")
    for name, expected_hash in metrics["adapter_sha256"].items():
        if Path(name).name != name or sha256((directory / "adapter" / name).read_bytes()) != expected_hash:
            raise ValueError("Saved adapter failed its recorded SHA256 check.")
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("Requires a CPU-only PyTorch build.")
    torch.set_num_threads(threads)
    options = {"revision": config["revision"], "local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(directory / "tokenizer", local_files_only=True, trust_remote_code=False)
    actual_label_tokens = [tokenizer.encode(label, add_special_tokens=False) for label in ("0", "1")]
    if actual_label_tokens != [[value] for value in config["label_token_ids"]]:
        raise ValueError("Saved label tokens do not match the loaded tokenizer.")
    base = AutoModelForCausalLM.from_pretrained(config["model_id"], torch_dtype=torch.float32, attn_implementation="sdpa", **options).to("cpu")
    model = PeftModel.from_pretrained(base, directory / "adapter", is_trainable=False)
    assert_cpu_model(model)
    encoded = encode_split(tokenizer, [{"id": 0, "text": text, "label": 0, "text_sha256": sha256(text.encode("utf-8"))}], config["max_sequence_length"])
    result = evaluate(model, encoded, tokenizer.pad_token_id, config["label_token_ids"], 1)["records"][0]
    return {"label": "spam" if result["predicted"] else "ham", "spam_score": result["spam_probability_conditioned_on_two_labels"], "message_truncated": result["message_truncated"], "device": "cpu", "score_is_calibrated_probability": False}


def predict_naive_bayes(directory: Path, text: str):
    status = json.loads((directory / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError("Only a completed training run can be loaded.")
    metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8"))
    if sha256((directory / "naive_bayes.json").read_bytes()) != metrics["naive_bayes_recipe_sha256"]:
        raise ValueError("Naive Bayes recipe failed its recorded SHA256 check.")
    recipe = json.loads((directory / "naive_bayes.json").read_text(encoding="utf-8"))
    counts = recipe["word_counts"]
    vocabulary = set(counts["0"]) | set(counts["1"])
    words = Counter(word for word in nb_tokens(text) if word in vocabulary)
    documents = recipe["class_document_counts"]
    scores = []
    for label in ("0", "1"):
        denominator = sum(counts[label].values()) + recipe["alpha"] * len(vocabulary)
        score = math.log(documents[label] / sum(documents.values()))
        score += sum(number * math.log((counts[label].get(word, 0) + recipe["alpha"]) / denominator) for word, number in words.items())
        scores.append(score)
    return {"label": "spam" if scores[1] > scores[0] else "ham", "method": "multinomial_naive_bayes", "device": "cpu"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/real-task.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--archive", type=Path, default=Path(".cache/real-task/sms-spam-collection.zip"))
    parser.add_argument("--download-data", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--predict")
    parser.add_argument("--adapter-run", type=Path)
    parser.add_argument("--method", choices=("lora", "naive-bayes"), default="lora")
    args = parser.parse_args()
    if args.predict is not None:
        if args.adapter_run is None:
            parser.error("--predict requires --adapter-run")
        prediction = predict_naive_bayes(args.adapter_run, args.predict) if args.method == "naive-bayes" else predict_local(args.adapter_run, args.predict, args.threads or 4)
        print(json.dumps(prediction))
        return
    if args.output is None:
        parser.error("--output is required for training")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key, value in (("optimizer_steps", args.steps), ("threads", args.threads)):
        if value is not None:
            config[key] = value
    train(config, args.output, args.archive, args.download_data)


if __name__ == "__main__":
    main()
