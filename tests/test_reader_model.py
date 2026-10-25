"""Offline reader boundaries and serialization, without task weights or datasets."""
import copy
import json

import pytest
import torch
from transformers import BertConfig, BertForQuestionAnswering, BertTokenizerFast

from cpu_ai_lab import reader_model as task


@pytest.fixture
def tokenizer(tmp_path):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "who", "is", "alice", "bob",
             "met", "in", "paris", "new", "york", "city", "?", ".", "naive", "playing", "play", "##ing"]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(vocab) + "\n", encoding="utf-8")
    return BertTokenizerFast(vocab_file=str(path), do_lower_case=True)


def span(context, text, start=None):
    start = context.index(text) if start is None else start
    return {"text": text, "start": start, "end": start + len(text)}


def test_full_context_offsets_and_exact_training_labels(tokenizer):
    context = "Alice met Bob in Paris."
    feature = task.encode_example(tokenizer, "Who is Bob?", context)
    first, last = task.training_positions(feature, context, [span(context, "Bob")], False)
    assert context[feature["offsets"][first][0]:feature["offsets"][last][1]] == "Bob"
    assert all(feature["context_mask"][first:last + 1])
    assert task.training_positions(feature, context, [], True) == (feature["cls_index"],) * 2
    assert all(feature["attention_mask"])


def test_complete_budget_rejects_instead_of_truncating(tokenizer):
    context = "Alice " * 10
    actual = task.encode_example(tokenizer, "Who?", context)
    assert task.encode_example(tokenizer, "Who?", context, len(actual["input_ids"])) == actual
    with pytest.raises(ValueError, match="truncation is forbidden"):
        task.encode_example(tokenizer, "Who?", context, len(actual["input_ids"]) - 1)
    with pytest.raises(ValueError, match="512"):
        task.encode_example(tokenizer, "Who?", context, 513)


def test_unaligned_or_overlong_positive_never_becomes_null(tokenizer):
    context = "Alice met Bob in New York City."
    feature = task.encode_example(tokenizer, "Who?", context)
    with pytest.raises(ValueError, match="no exact, aligned, bounded"):
        task.training_positions(feature, context, [span(context, "Ali")], False)
    with pytest.raises(ValueError, match="no exact, aligned, bounded"):
        task.training_positions(feature, context, [span(context, "New York City")], False, 2)
    with pytest.raises(ValueError, match="exactly match"):
        task.training_positions(feature, context, [{"start": 0, "end": 5, "text": "alice"}], False)
    with pytest.raises(ValueError, match="cannot have positive"):
        task.training_positions(feature, context, [span(context, "Alice")], True)


def test_multiple_labels_choose_fixed_earliest_span(tokenizer):
    context = "Bob met Alice in Paris."
    feature = task.encode_example(tokenizer, "Who?", context)
    labels = [span(context, "Alice"), span(context, "Bob")]
    result = task.training_positions(feature, context, labels, False)
    assert result == task.training_positions(feature, context, list(reversed(labels)), False)
    assert feature["offsets"][result[0]][0] == 0


def test_unicode_character_offsets_preserve_original_answer(tokenizer):
    context = "Alice is naïve."
    feature = task.encode_example(tokenizer, "Who?", context)
    first, last = task.training_positions(feature, context, [span(context, "naïve")], False)
    starts = [-10.0] * len(feature["input_ids"])
    ends = starts.copy()
    starts[first], ends[last] = 7.0, 8.0
    result = task.decode_span(context, feature, starts, ends)
    assert result["span_prediction"] == "naïve"
    assert result["null_minus_span_score"] == -35.0


def test_decoder_excludes_question_special_padding_and_invalid_endpoint_pairs(tokenizer):
    context = "Alice met Bob in Paris."
    feature = task.encode_example(tokenizer, "Who is Bob?", context)
    first, last = task.training_positions(feature, context, [span(context, "Bob")], False)
    size = len(feature["input_ids"])
    starts, ends = [-10.0] * size, [-10.0] * size
    # Huge question/special logits must not become answer spans.
    starts[1], ends[-1] = 1000.0, 1000.0
    # Separately best endpoints are reversed; the exact bounded search handles this.
    starts[last + 1], ends[first - 1] = 20.0, 20.0
    starts[first], ends[last] = 18.0, 18.0
    starts[feature["cls_index"]], ends[feature["cls_index"]] = 3.0, 4.0
    result = task.decode_span(context, feature, starts, ends, max_answer_tokens=1)
    assert result["span_prediction"] == "Bob"
    assert result["best_span_score"] == 36
    assert result["null_score"] == 7
    assert result["null_minus_span_score"] == -29


def test_decoder_maximum_span_length_and_tie_are_deterministic(tokenizer):
    context = "New York City met Alice."
    feature = task.encode_example(tokenizer, "Who?", context)
    first, last = task.training_positions(feature, context, [span(context, "New York City")], False)
    starts, ends = [-5.0] * len(feature["input_ids"]), [-5.0] * len(feature["input_ids"])
    starts[first], ends[last] = 20.0, 20.0
    assert task.decode_span(context, feature, starts, ends, 3)["span_prediction"] == "New York City"
    assert task.decode_span(context, feature, starts, ends, 2)["end_token"] - task.decode_span(context, feature, starts, ends, 2)["start_token"] < 2
    tied = task.decode_span(context, feature, [0.0] * len(starts), [0.0] * len(ends))
    assert tied["span_prediction"] == "New"
    assert tied["null_minus_span_score"] == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_decoder_rejects_nonfinite_logits(tokenizer, bad):
    feature = task.encode_example(tokenizer, "Who?", "Alice.")
    values = [0.0] * len(feature["input_ids"])
    values[0] = bad
    with pytest.raises(ValueError, match="finite scalar"):
        task.decode_span("Alice.", feature, values, [0.0] * len(values))


def test_collation_is_cpu_right_padding_with_unpadded_positions(tokenizer):
    features = [task.encode_example(tokenizer, "Who?", text) for text in ("Alice.", "Alice met Bob in Paris.")]
    labels = [(feature["cls_index"],) * 2 for feature in features]
    batch = task.collate(features, tokenizer.pad_token_id, labels)
    assert all(value.device.type == "cpu" for value in batch.values())
    assert batch["input_ids"][0, -1].item() == tokenizer.pad_token_id
    assert batch["attention_mask"][0, -1].item() == 0
    assert batch["end_positions"].tolist() == [0, 0]
    with pytest.raises(ValueError, match="outside its unpadded"):
        task.collate(features, tokenizer.pad_token_id, [(999, 999), (0, 0)])


def tiny_model(tokenizer):
    return BertForQuestionAnswering(BertConfig(vocab_size=len(tokenizer), hidden_size=16,
        num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
        max_position_embeddings=512, num_labels=2, hidden_dropout_prob=0,
        attention_probs_dropout_prob=0, pad_token_id=tokenizer.pad_token_id,
        attn_implementation="sdpa"))


def test_predict_is_gold_free_single_forward_and_restores_training_mode(tokenizer):
    torch.set_num_threads(1)
    model = tiny_model(tokenizer)
    model.train()
    rows = [{"question": "Who?", "context": "Alice met Bob.", "references": ["irrelevant"], "is_impossible": True}]
    calls = []
    hook = model.register_forward_hook(lambda *_: calls.append(1))
    original = task.predict_batch(model, tokenizer, rows)
    assert calls == [1] and model.training
    hook.remove()
    altered = copy.deepcopy(rows)
    altered[0].update(references=["Alice"], is_impossible=False)
    assert original == task.predict_batch(model, tokenizer, altered)


def test_model_and_tokenizer_round_trip_and_corruption_detection(tokenizer, tmp_path):
    torch.set_num_threads(1)
    model = tiny_model(tokenizer)
    rows = [{"question": "Who?", "context": "Alice met Bob in Paris."}]
    expected = task.predict_batch(model, tokenizer, rows)
    path = tmp_path / "saved-reader"
    manifest = task.save_reader(path, model, tokenizer, {"step": 2})
    loaded, loaded_tokenizer = task.load_reader(path)
    actual = task.predict_batch(loaded, loaded_tokenizer, rows)
    for key, value in expected[0].items():
        assert actual[0][key] == (pytest.approx(value, abs=1e-6) if type(value) is float else value)
    assert all(torch.equal(value, loaded.state_dict()[key]) for key, value in model.state_dict().items())
    assert manifest["metadata"] == {"step": 2}
    with pytest.raises(FileExistsError, match="Preserving"):
        task.save_reader(path, model, tokenizer)
    config_path = path / "config.json"
    config_path.write_text(config_path.read_text() + " ", encoding="utf-8")
    with pytest.raises(ValueError, match="file integrity"):
        task.load_reader(path)


def test_manifest_mutation_is_detected_before_loading(tokenizer, tmp_path):
    model = tiny_model(tokenizer)
    path = tmp_path / "saved-reader"
    task.save_reader(path, model, tokenizer)
    manifest_path = path / "reader_manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["max_answer_tokens"] = 999
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest integrity"):
        task.load_reader(path)


def test_initializer_rejects_unpinned_weights_without_loading(tmp_path):
    (tmp_path / "model.safetensors").write_bytes(b"not the publisher weights")
    with pytest.raises(ValueError, match="pinned publisher"):
        task.load_initial(tmp_path, seed=31)


def test_unbound_checkpoint_file_is_rejected(tokenizer, tmp_path):
    path = tmp_path / "saved-reader"
    task.save_reader(path, tiny_model(tokenizer), tokenizer)
    (path / "unexpected.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="Unbound"):
        task.load_reader(path)


def test_numpy_logits_are_supported(tokenizer):
    import numpy as np
    feature = task.encode_example(tokenizer, "Who?", "Alice.")
    logits = np.zeros(len(feature["input_ids"]), dtype=np.float32)
    assert task.decode_span("Alice.", feature, logits, logits)["span_prediction"] == "Alice"
