"""Objective/gradient equivalence with dropout disabled; no pretrained weights."""
import copy

import pytest
import torch
from transformers import BertConfig, BertForQuestionAnswering

from cpu_ai_lab import reader_model
from cpu_ai_lab.reader_training import backward_batch


def fixture():
    torch.manual_seed(51)
    model = BertForQuestionAnswering(BertConfig(vocab_size=32, hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32, max_position_embeddings=512,
        hidden_dropout_prob=0, attention_probs_dropout_prob=0, pad_token_id=0,
        num_labels=2, attn_implementation="sdpa"))
    lengths = [13, 5, 9, 7, 11, 4, 8, 6]
    features = [{"input_ids": [1] + [3 + index] * (size - 2) + [2],
                 "attention_mask": [1] * size, "token_type_ids": [0, 0] + [1] * (size - 2)}
                for index, size in enumerate(lengths)]
    positions = [(0, 0) if index % 2 else (2, 3) for index in range(len(features))]
    return model, features, positions


@pytest.mark.parametrize("microbatch", [1, 2, 3, 4, 8, 20])
def test_weighted_microbatch_loss_and_gradients_match_full_global_batch(microbatch):
    torch.set_num_threads(1)
    full, features, positions = fixture()
    micro = copy.deepcopy(full)
    indices = [6, 0, 5, 3, 1, 7, 4, 2]
    batch = reader_model.collate([features[i] for i in indices], 0, [positions[i] for i in indices])
    expected = full(**batch).loss
    expected.backward()
    result = backward_batch(micro, features, positions, indices, 0, microbatch)
    assert result["mean_loss"] == pytest.approx(float(expected.detach()), abs=5e-7)
    for (name, left), (other_name, right) in zip(full.named_parameters(), micro.named_parameters(), strict=True):
        assert name == other_name
        assert torch.allclose(left.grad, right.grad, atol=2e-6, rtol=2e-5), name
    assert result["examples"] == 8 and result["input_tokens"] == sum(len(row["input_ids"]) for row in features)
    assert result["padded_tokens"] == batch["input_ids"].numel()
    assert result["global_padded_width"] == batch["input_ids"].shape[1]
    assert sorted(result["ordered_indices"]) == sorted(indices)
    assert sum(result["microbatch_sizes"]) == 8
    assert result["ordered_indices"] == sorted(indices, key=lambda i: (len(features[i]["input_ids"]), i))


def test_helper_accumulates_existing_gradients_without_updating_or_zeroing():
    torch.set_num_threads(1)
    model, features, positions = fixture()
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    backward_batch(model, features, positions, list(range(8)), 0, 4)
    once = {name: parameter.grad.clone() for name, parameter in model.named_parameters()}
    backward_batch(model, features, positions, list(range(8)), 0, 4)
    for name, parameter in model.named_parameters():
        assert torch.equal(before[name], parameter.detach())
        assert torch.allclose(parameter.grad, once[name] * 2, atol=2e-6, rtol=2e-5)


@pytest.mark.parametrize("indices,size", [([], 2), ([0, 0], 2), ([99], 2), ([True], 2), ([0], 0)])
def test_invalid_global_batch_or_microbatch_rejected(indices, size):
    model, features, positions = fixture()
    with pytest.raises(ValueError):
        backward_batch(model, features, positions, indices, 0, size)
