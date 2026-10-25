"""CPU extractive reader with complete inputs, exact labels and gold-free decoding.

The initializer already received SQuAD2 supervision; this module does not claim
that its previous training was CPU-only. New adaptation and inference are CPU
operations. Threshold selection belongs to reader_metrics, not this decoder.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


MODEL_ID = "deepset/minilm-uncased-squad2"
REVISION = "934656cdda79824eabf503ed56e15c01ddbdbe3f"
MODEL_LICENSE = "CC-BY-4.0"
INITIAL_WEIGHTS_SHA256 = "240478c7251d5e55be35feb47177054b960259b9328af662ea3cee87ce9755a6"
MAX_LENGTH = 512
MAX_ANSWER_TOKENS = 32
FORMAT = "cpu-extractive-reader-v1"
INPUT_KEYS = ("input_ids", "attention_mask", "token_type_ids")


def _positive_int(value, name):
    if type(value) is not int or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


def encode_example(tokenizer, question: str, context: str, max_length: int = MAX_LENGTH) -> dict:
    """Encode one complete question/context pair; no truncation or gold inputs."""
    _positive_int(max_length, "max_length")
    if max_length > MAX_LENGTH:
        raise ValueError("The reader supports at most 512 total tokens.")
    if not getattr(tokenizer, "is_fast", False):
        raise ValueError("Exact offset alignment requires a fast tokenizer.")
    if not isinstance(question, str) or not question.strip() or not isinstance(context, str) or not context.strip():
        raise ValueError("Question and complete context must be nonempty strings.")
    batch = tokenizer(question, context, add_special_tokens=True, truncation=False,
                      padding=False, return_offsets_mapping=True)
    if len(batch["input_ids"]) > max_length:
        raise ValueError("Complete question/context exceeds the reader token budget; truncation is forbidden.")
    sequence_ids = batch.sequence_ids()
    offsets = [list(pair) for pair in batch["offset_mapping"]]
    context_mask = [sequence == 1 and end > start for sequence, (start, end) in zip(sequence_ids, offsets, strict=True)]
    special_cls = [index for index, (token, sequence) in enumerate(zip(batch["input_ids"], sequence_ids, strict=True))
                   if token == tokenizer.cls_token_id and sequence is None]
    if len(special_cls) != 1 or not any(context_mask):
        raise ValueError("Reader encoding requires one special CLS and nonempty context offsets.")
    result = {key: list(batch[key]) for key in INPUT_KEYS if key in batch}
    result.update(offsets=offsets, context_mask=context_mask, cls_index=special_cls[0])
    _validate_encoding(result, context)
    return result


def _validate_encoding(encoded, context):
    size = len(encoded["input_ids"])
    if not 1 <= size <= MAX_LENGTH or any(len(encoded[key]) != size for key in ("attention_mask", "offsets", "context_mask")):
        raise ValueError("Invalid reader feature lengths.")
    if "token_type_ids" in encoded and len(encoded["token_type_ids"]) != size:
        raise ValueError("Invalid segment feature length.")
    cls = encoded["cls_index"]
    if type(cls) is not int or not 0 <= cls < size or encoded["context_mask"][cls]:
        raise ValueError("Invalid CLS null position.")
    previous_end = 0
    for index, (offset, is_context) in enumerate(zip(encoded["offsets"], encoded["context_mask"], strict=True)):
        if type(is_context) is not bool or len(offset) != 2 or any(type(value) is not int for value in offset):
            raise ValueError("Malformed token offsets or context mask.")
        if is_context:
            start, end = offset
            if encoded["attention_mask"][index] != 1 or not 0 <= start < end <= len(context) or start < previous_end:
                raise ValueError("Context offsets must be ordered, attended and within the complete context.")
            previous_end = end
    if encoded["attention_mask"][cls] != 1:
        raise ValueError("CLS null position must be attended.")


def training_positions(encoded: dict, context: str, literal_answer_spans: list[dict],
                       is_impossible: bool, max_answer_tokens: int = MAX_ANSWER_TOKENS) -> tuple[int, int]:
    """Map native negatives to CLS; never silently turn unaligned positives into null."""
    _positive_int(max_answer_tokens, "max_answer_tokens")
    _validate_encoding(encoded, context)
    if type(is_impossible) is not bool:
        raise ValueError("is_impossible must be a boolean.")
    if is_impossible:
        if literal_answer_spans:
            raise ValueError("Native missing questions cannot have positive span labels.")
        return encoded["cls_index"], encoded["cls_index"]
    candidates = []
    for span in literal_answer_spans:
        start, end, text = span["start"], span["end"], span["text"]
        if type(start) is not int or type(end) is not int or not isinstance(text, str) or not text.strip():
            raise ValueError("Malformed literal reference span.")
        if not 0 <= start < end <= len(context) or context[start:end] != text:
            raise ValueError("Reference span must exactly match the original context.")
        starts = [i for i, (offset, mask) in enumerate(zip(encoded["offsets"], encoded["context_mask"], strict=True)) if mask and offset[0] == start]
        ends = [i for i, (offset, mask) in enumerate(zip(encoded["offsets"], encoded["context_mask"], strict=True)) if mask and offset[1] == end]
        for first in starts:
            for last in ends:
                if first <= last and last - first + 1 <= max_answer_tokens and all(encoded["context_mask"][first:last + 1]):
                    candidates.append((start, end, first, last))
    if not candidates:
        raise ValueError("Answerable question has no exact, aligned, bounded reference span.")
    # Earliest character span, then earliest end: a fixed target independent of logits.
    _, _, first, last = min(candidates)
    return first, last


def _logits(values, size):
    device = getattr(values, "device", None)
    if device is not None and getattr(device, "type", device) != "cpu":
        raise ValueError("Reader logits must remain on CPU.")
    if hasattr(values, "tolist"):
        values = values.tolist()
    if len(values) != size or any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("Expected one finite scalar logit per input token.")
    return values


def decode_span(context: str, encoded: dict, start_logits, end_logits,
                max_answer_tokens: int = MAX_ANSWER_TOKENS) -> dict:
    """Exactly search all bounded context spans; return raw null-minus-span score.

No top-k pruning: the best valid span may have neither globally maximal endpoint.
Equal scores select the earlier start then earlier end. Scores are logits, not
calibrated probabilities. No answerability label or reference enters decoding.
"""
    _positive_int(max_answer_tokens, "max_answer_tokens")
    _validate_encoding(encoded, context)
    size = len(encoded["input_ids"])
    starts, ends = _logits(start_logits, size), _logits(end_logits, size)
    best = None
    for first in range(size):
        if not encoded["context_mask"][first]:
            continue
        for last in range(first, min(size, first + max_answer_tokens)):
            if not encoded["context_mask"][last]:
                break
            char_start, char_end = encoded["offsets"][first][0], encoded["offsets"][last][1]
            if not context[char_start:char_end].strip():
                continue
            score = float(starts[first] + ends[last])
            if not math.isfinite(score):
                raise ValueError("Combined span logits must remain finite.")
            if best is None or score > best[0]:
                best = (score, first, last, char_start, char_end)
    if best is None:
        raise ValueError("Reader encoding has no valid nonempty context span.")
    score, first, last, char_start, char_end = best
    cls = encoded["cls_index"]
    null = float(starts[cls] + ends[cls])
    if not math.isfinite(null) or not math.isfinite(null - score):
        raise ValueError("Combined null logits and score difference must remain finite.")
    return {"span_prediction": context[char_start:char_end], "best_span_score": score,
            "null_score": null, "null_minus_span_score": null - score,
            "start_token": first, "end_token": last, "start_char": char_start, "end_char": char_end,
            "input_tokens": size}


def collate(features: list[dict], pad_token_id: int, positions=None) -> dict:
    """Right-pad model inputs on CPU; offsets and all gold metadata stay outside."""
    import torch
    if not features or type(pad_token_id) is not int or pad_token_id < 0:
        raise ValueError("A nonempty feature batch and valid pad token are required.")
    if positions is not None and len(positions) != len(features):
        raise ValueError("One start/end label pair is required per training example.")
    width = max(len(feature["input_ids"]) for feature in features)
    if width > MAX_LENGTH:
        raise ValueError("Reader batch exceeds the model position budget.")
    keys = [key for key in INPUT_KEYS if key in features[0]]
    if any(set(key for key in INPUT_KEYS if key in feature) != set(keys) for feature in features):
        raise ValueError("All batch members must have the same model input keys.")
    result = {key: torch.tensor([feature[key] + [pad_token_id if key == "input_ids" else 0] * (width - len(feature[key]))
                                for feature in features], dtype=torch.long, device="cpu") for key in keys}
    if positions is not None:
        for feature, pair in zip(features, positions, strict=True):
            if len(pair) != 2 or any(type(value) is not int or not 0 <= value < len(feature["input_ids"]) for value in pair):
                raise ValueError("Training position is outside its unpadded input.")
        result["start_positions"] = torch.tensor([pair[0] for pair in positions], dtype=torch.long, device="cpu")
        result["end_positions"] = torch.tensor([pair[1] for pair in positions], dtype=torch.long, device="cpu")
    return result


def assert_cpu(model):
    if any(value.device.type != "cpu" for value in list(model.parameters()) + list(model.buffers())):
        raise RuntimeError("The extractive reader must stay entirely on CPU.")


def predict_batch(model, tokenizer, rows: list[dict], max_length: int = MAX_LENGTH,
                  max_answer_tokens: int = MAX_ANSWER_TOKENS) -> list[dict]:
    """One forward pass; only question/context are read from each input record."""
    import torch
    assert_cpu(model)
    if not rows:
        return []
    features = [encode_example(tokenizer, row["question"], row["context"], max_length) for row in rows]
    inputs = collate(features, tokenizer.pad_token_id)
    was_training = model.training
    model.eval()
    try:
        with torch.inference_mode():
            output = model(**inputs)
        return [decode_span(row["context"], feature,
                            output.start_logits[index, :len(feature["input_ids"])],
                            output.end_logits[index, :len(feature["input_ids"])], max_answer_tokens)
                for index, (row, feature) in enumerate(zip(rows, features, strict=True))]
    finally:
        model.train(was_training)


def _load_local(path):
    import torch
    from transformers import AutoModelForQuestionAnswering, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(path, use_fast=True, local_files_only=True, trust_remote_code=False)
    if not tokenizer.is_fast:
        raise ValueError("Saved reader tokenizer must support exact character offsets.")
    model = AutoModelForQuestionAnswering.from_pretrained(path, local_files_only=True, trust_remote_code=False,
                                                        use_safetensors=True, torch_dtype=torch.float32,
                                                        attn_implementation="sdpa")
    model.to("cpu")
    assert_cpu(model)
    if model.config.model_type != "bert" or model.config.num_labels != 2 or model.config.max_position_embeddings != MAX_LENGTH:
        raise ValueError("Reader architecture must be BERT with two span logits and 512 positions.")
    model.eval()
    return model, tokenizer


def _hash(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def load_initial(path, *, seed: int = 0):
    """Load the pinned QA initializer already downloaded by the orchestrator."""
    import torch
    if type(seed) is not int:
        raise ValueError("A deterministic integer seed is required.")
    if _hash(Path(path) / "model.safetensors") != INITIAL_WEIGHTS_SHA256:
        raise ValueError("Initializer weights differ from the pinned publisher artifact.")
    torch.manual_seed(seed)
    return _load_local(path)


def save_reader(path, model, tokenizer, metadata=None) -> dict:
    """Save weights/tokenizer once, with hashes checked before every later load."""
    path = Path(path)
    if path.exists():
        raise FileExistsError("Preserving the existing reader checkpoint.")
    assert_cpu(model)
    # Validate JSON before creating an output directory.
    metadata = json.loads(json.dumps(metadata if metadata is not None else {}, allow_nan=False))
    path.mkdir(parents=True)
    model.save_pretrained(path, safe_serialization=True)
    tokenizer.save_pretrained(path)
    manifest = {"format": FORMAT, "initializer": {"model_id": MODEL_ID, "revision": REVISION,
                "license": MODEL_LICENSE, "weights_sha256": INITIAL_WEIGHTS_SHA256},
                "max_length": MAX_LENGTH, "max_answer_tokens": MAX_ANSWER_TOKENS,
                "score": "null_minus_span_score; larger values favor abstention",
                "metadata": metadata,
                "files_sha256": {file.name: _hash(file) for file in sorted(path.iterdir()) if file.is_file()}}
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    manifest["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    (path / "reader_manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return manifest


def load_reader(path):
    path = Path(path)
    manifest = json.loads((path / "reader_manifest.json").read_bytes())
    expected = manifest.pop("manifest_sha256")
    payload = json.dumps(manifest, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    if hashlib.sha256(payload).hexdigest() != expected or manifest["format"] != FORMAT:
        raise ValueError("Reader manifest integrity check failed.")
    if manifest["max_length"] != MAX_LENGTH or manifest["max_answer_tokens"] != MAX_ANSWER_TOKENS:
        raise ValueError("Reader decoding contract differs from this implementation.")
    files = manifest["files_sha256"]
    if not {"config.json", "model.safetensors"}.issubset(files) or not ({"tokenizer.json", "vocab.txt"} & files.keys()):
        raise ValueError("Reader manifest must bind the model and tokenizer files.")
    actual_files = {file.name for file in path.iterdir() if file.is_file() and file.name != "reader_manifest.json"}
    if actual_files != set(files):
        raise ValueError("Unbound or missing files in the reader checkpoint.")
    for name, digest in files.items():
        if Path(name).name != name or "/" in name or "\\" in name or name in (".", ".."):
            raise ValueError("Reader manifest file names must remain within the checkpoint.")
        if not (path / name).is_file() or _hash(path / name) != digest:
            raise ValueError("Saved reader file integrity check failed.")
    return _load_local(path)
