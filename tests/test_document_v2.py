import copy
import json
from pathlib import Path

import pytest
import torch

from cpu_ai_lab import document_v2 as task


def rows():
    result = []
    for number in range(3):
        for missing in (False, True):
            for length in ("short", "long"):
                result.append({"id": f"{missing}-{length}-{number}", "is_impossible": missing,
                               "length_group": length, "context_tokens": 10 if length == "short" else 250,
                               "context": "Paris in France", "question": "Which city?",
                               "references": [] if missing else ["Paris"], "prompt": "6" if missing else "9 5"})
    return result


def test_stratified_schedule_is_reproducible_and_covers_one_epoch():
    examples = rows()
    schedule = task.sample_schedule(examples, 3, 42)
    assert schedule == task.sample_schedule(examples, 3, 42)
    assert sorted(i for step in schedule for i in step) == list(range(12))
    for step in schedule:
        assert len({task.data.stratum(examples[i]) for i in step}) == 4
    extended = task.sample_schedule(examples, 5, 42)
    assert extended[:3] == schedule
    assert all(len(set(step)) == 4 for step in extended)
    with pytest.raises(ValueError, match="stratum"):
        task.sample_schedule(examples[:1], 1, 42)


def test_generation_restores_order_and_distinguishes_missing_answers(monkeypatch):
    monkeypatch.setattr(task, "render_prompt", lambda tokenizer, row: row["prompt"])

    class Tokenizer:
        pad_token_id, eos_token_id = 0, 2
        def encode(self, text, **kwargs):
            return [int(value) for value in text.split()]
        def decode(self, ids, **kwargs):
            return " ".join({11: "Paris", 12: "NOT_FOUND"}[i] for i in ids if i not in (0, 2))

    class Model:
        def eval(self):
            return self
        def generate(self, input_ids, attention_mask, **kwargs):
            assert kwargs["do_sample"] is False
            outputs = torch.tensor([[12 if int(ids[-1]) == 6 else 11, 2, 0] for ids in input_ids])
            return torch.cat((input_ids, outputs), dim=1)

    examples = rows()[:4]
    result, local = task.generation_evaluation(Model(), Tokenizer(), examples, 3, 3)
    assert [r["id"] for r in result["records"]] == [r["id"] for r in examples]
    assert result["exact_match"] == result["token_f1"] == result["harmonic_utility"] == 1
    assert result["generated_tokens_including_eos"] == 8
    assert result["ended_with_eos"] == 4
    assert result["coverage"] == 0.5
    assert all("prediction" not in r for r in result["records"])
    assert all("prediction" in r for r in local)


def test_configuration_and_state_hash_guards():
    config = json.loads(Path("configs/document-v2.json").read_text())
    task.validate_config(config)
    for key, value in (("batch_size", True), ("batch_size", 2), ("learning_rate", True),
                       ("learning_rate", float("nan")), ("training_threads", 0)):
        invalid = {**config, key: value}
        with pytest.raises(ValueError):
            task.validate_config(invalid)
    first = {"b": torch.tensor([2.0]), "a": torch.tensor([1.0])}
    second = {"a": first["a"].clone(), "b": first["b"].clone()}
    assert task.state_hash(first) == task.state_hash(second)
    second["a"].add_(1)
    assert task.state_hash(first) != task.state_hash(second)
    assert task.state_change(second, first) == 1


def test_prediction_rejects_long_context_without_silent_truncation(monkeypatch):
    monkeypatch.setattr(task.data, "prompt_text", lambda context, question: context)
    monkeypatch.setattr(task, "render_prompt", lambda tokenizer, row: row["prompt"])
    class Tokenizer:
        def encode(self, text, **kwargs):
            return list(range(len(text.split())))
        def __call__(self, text, **kwargs):
            return {"input_ids": torch.tensor([self.encode(text)])}
    with pytest.raises(ValueError, match="no automatic truncation"):
        task.answer(None, Tokenizer(), {"max_context_tokens": 2, "max_prompt_tokens": 10}, "three word passage", "which?")
