"""Task-specific split, classification, and CPU objective tests without weights."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import pytest
import torch

from cpu_ai_lab.real_task import (
    classification_metrics, collate_classification, deduplicate, encode_message,
    label_logits, load_sms, make_splits, naive_bayes, normalize_template,
    predict_naive_bayes, predict_local, sha256,
)


def row(index, text, label):
    return {"id": index, "text": text, "label": label, "text_sha256": str(index)}


def test_near_duplicate_templates_cannot_cross_splits():
    examples = [
        row(1, "Win a FREE phone today call 12345 to claim your amazing prize now", 1),
        row(2, "win a free phone today call 99999 to claim your amazing prize now!", 1),
        row(3, "Win a free phone today call 54321 to claim your amazing prize now please", 1),
        row(4, "Are we meeting for lunch?", 0),
    ]
    kept, report = deduplicate(examples)
    assert len(kept) == 2
    assert kept[0]["duplicate_group_size"] == 3
    assert report["duplicate_or_conflicting_rows_removed"] == 2


def test_conflicting_duplicate_labels_removed_entirely():
    kept, report = deduplicate([row(1, "Meet at 123", 0), row(2, "Meet at 456", 1)])
    assert not kept
    assert report["conflicting_label_rows_removed"] == 2


def test_template_normalizes_contact_variants():
    first = normalize_template("Visit https://test.example/x or e-mail a@example.com call 123.")
    second = normalize_template("VISIT https://else.example/y or e-mail b@example.net call 456!")
    assert first == second


def test_stratified_splits_disjoint_and_test_stable_when_training_size_changes():
    examples = [row(i, str(i), int(i % 5 == 0)) for i in range(100)]
    smaller = make_splits(examples, 30, 10, 20, 7)
    larger = make_splits(examples, 50, 10, 20, 7)
    assert smaller["test"] == larger["test"]
    assert smaller["validation"] == larger["validation"]
    ids = [item["id"] for values in larger.values() for item in values]
    assert len(ids) == len(set(ids))
    assert sum(item["label"] for item in smaller["test"]) == 4
    with pytest.raises(ValueError):
        make_splits(examples, 80, 20, 20, 7)


def test_confusion_matrix_and_macro_f1_show_minority_failure():
    metric = classification_metrics([0, 0, 0, 1], [0, 0, 0, 0])
    assert metric["accuracy"] == 0.75
    assert metric["confusion_matrix_true_rows_predicted_columns_ham_spam"] == [[3, 0], [1, 0]]
    assert metric["per_class"]["spam"]["recall"] == 0
    assert metric["macro_f1"] == pytest.approx(3 / 7)
    with pytest.raises(ValueError):
        classification_metrics([0, 1], [0, 3])


def test_nb_training_vocabulary_ignores_test_words_and_recipe_reproduces():
    splits = {
        "train": [row(1, "lunch meet friend", 0), row(2, "free prize winner", 1)],
        "validation": [row(3, "friend lunch", 0), row(4, "prize winner", 1)],
        "test": [row(5, "friend lunch secretword", 0), row(6, "free prize secretword", 1)],
    }
    with tempfile.TemporaryDirectory() as name:
        recipe = Path(name) / "naive_bayes.json"
        result = naive_bayes(splits, recipe_output=recipe)
        (Path(name) / "run_status.json").write_text(json.dumps({"status": "complete"}))
        (Path(name) / "metrics.json").write_text(json.dumps({"naive_bayes_recipe_sha256": sha256(recipe.read_bytes())}))
        assert result["vocabulary_size"] == 6
        assert result["test"]["accuracy"] == 1
        assert predict_naive_bayes(Path(name), "free prize")["label"] == "spam"
        assert "secretword" not in recipe.read_text()


def test_loader_rejects_incomplete_run_before_loading_any_model():
    with tempfile.TemporaryDirectory() as name:
        path = Path(name)
        (path / "run_status.json").write_text(json.dumps({"status": "running"}))
        with pytest.raises(ValueError, match="completed"):
            predict_local(path, "hello")
        with pytest.raises(ValueError, match="completed"):
            predict_naive_bayes(path, "hello")


def test_left_padding_last_position_and_rotary_positions_are_correct():
    batch, labels = collate_classification([
        {"input_ids": [2, 3, 4], "label": 0}, {"input_ids": [5], "label": 1},
    ], 9)
    assert batch["input_ids"].tolist() == [[2, 3, 4], [9, 9, 5]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [0, 0, 1]]
    assert batch["position_ids"].tolist() == [[0, 1, 2], [0, 0, 0]]
    assert labels.tolist() == [0, 1]
    assert all(tensor.device.type == "cpu" for tensor in batch.values())


def test_single_position_logits_preserve_training_gradient():
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.tensor([[[0.2, 0.6, 0.9, 0.1]]]))

        def forward(self, **kwargs):
            assert kwargs["logits_to_keep"] == 1
            assert kwargs["use_cache"] is False
            return SimpleNamespace(logits=self.weight)

    model = Model()
    logits = label_logits(model, {}, [1, 3])
    torch.nn.functional.cross_entropy(logits, torch.tensor([1])).backward()
    assert model.weight.grad[0, 0, 1] > 0
    assert model.weight.grad[0, 0, 3] < 0
    assert model.weight.grad[0, 0, 0] == 0


def test_archive_checksum_enforced_before_parse():
    with tempfile.TemporaryDirectory() as name:
        path = Path(name) / "wrong.zip"
        path.write_bytes(b"not the dataset")
        with pytest.raises(ValueError, match="SHA256"):
            load_sms(path)


def test_truncation_preserves_prompt_suffix_and_is_reported():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "<user>" + messages[0]["content"] + "<assistant>"

        def encode(self, text, **kwargs):
            return [ord(char) for char in text]

        def decode(self, ids, **kwargs):
            return "".join(chr(value) for value in ids)

    tokenizer = Tokenizer()
    ids, shortened = encode_message(tokenizer, "x" * 500, 200)
    assert shortened
    assert len(ids) <= 200
    assert tokenizer.decode(ids).endswith("<assistant>")
    assert tokenizer.decode(ids).startswith("<user>Classify the SMS")
    with pytest.raises(ValueError, match="instruction"):
        encode_message(tokenizer, "x" * 100, 5)
