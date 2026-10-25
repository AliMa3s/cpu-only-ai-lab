"""Deterministic, synthetic extraction data with disjoint held-out values.

The default task is deliberately narrow: copy two integer fields from a record
into JSON. It measures task adaptation, not general language understanding.
No external dataset or personal information is used.
"""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


def make_extraction_splits(
    train_size: int = 192,
    validation_size: int = 32,
    test_size: int = 32,
    seed: int = 42,
) -> dict[str, list[dict]]:
    sizes = {"train": train_size, "validation": validation_size, "test": test_size}
    if any(not isinstance(size, int) or size < 1 for size in sizes.values()):
        raise ValueError("Every split must contain at least one example.")
    total = sum(sizes.values())
    if total * 2 > 9000:
        raise ValueError("At most 4500 records fit the disjoint four-digit value pool.")
    rng = random.Random(seed)
    values = rng.sample(range(1000, 10000), total * 2)
    templates = (
        "Record: user={user}; ticket={ticket}; status={status}.",
        "Ticket {ticket} was opened by user {user}. Status: {status}.",
        "Status: {status}. The user number is {user} and the ticket number is {ticket}.",
    )
    instruction = "Extract user and ticket as integers. Reply only with JSON containing the keys user and ticket.\n"
    result: dict[str, list[dict]] = {}
    cursor = 0
    for split, size in sizes.items():
        examples = []
        for index in range(size):
            user, ticket = values[cursor * 2:cursor * 2 + 2]
            status = rng.choice(("open", "closed", "pending"))
            template = templates[index % len(templates)]
            examples.append({
                "id": f"{split}-{index:04d}",
                "prompt": instruction + template.format(user=user, ticket=ticket, status=status),
                "response": json.dumps({"user": user, "ticket": ticket}, separators=(",", ":")),
                "fields": {"user": user, "ticket": ticket},
            })
            cursor += 1
        rng.shuffle(examples)
        result[split] = examples
    validate_splits(result, require_disjoint_values=True)
    return result


def validate_splits(splits: dict[str, list[dict]], require_disjoint_values: bool = False) -> None:
    seen_prompts: set[str] = set()
    seen_values: set[int] = set()
    for name in ("train", "validation", "test"):
        examples = splits.get(name)
        if not examples:
            raise ValueError(f"Missing or empty {name} split.")
        prompts = set()
        values = set()
        for example in examples:
            if not isinstance(example.get("prompt"), str) or not example["prompt"].strip():
                raise ValueError(f"{name}: each prompt must be a nonempty string.")
            if not isinstance(example.get("response"), str) or not example["response"].strip():
                raise ValueError(f"{name}: each response must be a nonempty string.")
            if example["prompt"] in prompts:
                raise ValueError(f"Duplicate prompt within {name}.")
            prompts.add(example["prompt"])
            if require_disjoint_values:
                values.update(example["fields"].values())
        if prompts & seen_prompts:
            raise ValueError(f"Prompt leakage detected in {name}.")
        if require_disjoint_values and values & seen_values:
            raise ValueError(f"Field-value leakage detected in {name}.")
        seen_prompts.update(prompts)
        seen_values.update(values)


def load_splits(directory: Path) -> dict[str, list[dict]]:
    """Read explicit train/validation/test JSONL files, never auto-split user data."""
    splits = {}
    for split in ("train", "validation", "test"):
        path = directory / f"{split}.jsonl"
        with path.open(encoding="utf-8") as handle:
            splits[split] = [json.loads(line) for line in handle if line.strip()]
    validate_splits(splits)
    return splits


def save_splits(splits: dict[str, list[dict]], directory: Path) -> dict[str, str]:
    directory.mkdir(parents=True, exist_ok=True)
    checksums = {}
    for name, examples in splits.items():
        payload = "".join(json.dumps(example, sort_keys=True, ensure_ascii=False) + "\n" for example in examples)
        path = directory / f"{name}.jsonl"
        path.write_text(payload, encoding="utf-8", newline="\n")
        checksums[name] = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return checksums


def score_response(text: str, expected: str) -> dict[str, bool]:
    """Strict JSON parsing; no forgiving extraction of a JSON substring."""
    def reject_constant(value: str):
        raise ValueError(f"Non-JSON numeric constant: {value}")

    def unique_object(pairs: list[tuple]) -> dict:
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"Duplicate JSON object key: {key}")
            result[key] = value
        return result

    stripped = text.strip()
    result = {"text_exact": stripped == expected.strip(), "valid_json": False, "json_exact": False}
    try:
        parsed = json.loads(stripped, parse_constant=reject_constant, object_pairs_hook=unique_object)
        result["valid_json"] = True
        target = json.loads(expected, parse_constant=reject_constant, object_pairs_hook=unique_object)
        # bool equals int in Python; type-aware serialization avoids counting it.
        result["json_exact"] = json.dumps(parsed, sort_keys=True) == json.dumps(target, sort_keys=True)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return result
