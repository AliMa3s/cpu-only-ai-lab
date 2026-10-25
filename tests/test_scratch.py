import math

import pytest
import torch

from cpu_ai_lab.scratch import ByteTransformer, evaluate, random_batch, split_corpus, state_checksum, unigram_baseline


def tiny_config():
    return {"width": 16, "heads": 2, "layers": 2, "block_size": 8, "dropout": 0.1}


def test_split_positions_are_disjoint_and_complete():
    payload = bytes(range(200))
    splits, bounds = split_corpus(payload, 0.8, 0.1)
    assert bounds["train"]["stop_byte_exclusive"] == bounds["validation"]["start_byte_inclusive"]
    assert bounds["validation"]["stop_byte_exclusive"] == bounds["test"]["start_byte_inclusive"]
    assert b"".join(bytes(splits[name].tolist()) for name in ("train", "validation", "test")) == payload
    assert not (set(splits["train"].tolist()) & set(splits["test"].tolist()))


def test_batches_never_read_outside_their_split():
    segment = torch.arange(50, 90)
    x, y = random_batch(segment, 64, 8, torch.Generator().manual_seed(10))
    assert int(x.min()) >= 50
    assert int(y.max()) <= 89
    assert torch.equal(y, x + 1)


def test_future_bytes_cannot_change_earlier_logits():
    torch.manual_seed(12)
    model = ByteTransformer(tiny_config()).eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6]])
    second = torch.tensor([[1, 2, 3, 90, 91, 92]])
    with torch.inference_mode():
        assert torch.allclose(model(first)[:, :3], model(second)[:, :3], atol=1e-6, rtol=1e-6)


def test_eval_scores_every_target_once_including_partial_tail():
    class UniformModel:
        block_size = 8

        def eval(self):
            return self

        def __call__(self, x):
            return torch.zeros((*x.shape, 256))

    result = evaluate(UniformModel(), torch.arange(24), batch_size=2)
    assert result["target_bytes"] == 23
    assert result["loss_nats_per_byte"] == pytest.approx(math.log(256), abs=1e-6)
    assert result["byte_perplexity"] == pytest.approx(256, abs=1e-3)


def test_optimizer_updates_random_weights():
    torch.manual_seed(42)
    model = ByteTransformer(tiny_config())
    before = state_checksum(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    ids = torch.arange(8)[None, :]
    torch.nn.functional.cross_entropy(model(ids).reshape(-1, 256), (ids + 1).reshape(-1)).backward()
    optimizer.step()
    assert before != state_checksum(model)
    assert all(p.device.type == "cpu" for p in model.parameters())


def test_unigram_is_fitted_only_on_training_bytes():
    train = torch.tensor([7] * 100)
    common = unigram_baseline(train, torch.tensor([7, 7, 7]))
    unseen = unigram_baseline(train, torch.tensor([8, 8, 8]))
    assert common["loss_nats_per_byte"] < unseen["loss_nats_per_byte"]


@pytest.mark.parametrize("train,val", [(1, 0.1), (0, 0.1), (0.9, 0.2), (-0.1, 0.1)])
def test_invalid_splits_rejected(train, val):
    with pytest.raises(ValueError):
        split_corpus(b"x" * 100, train, val)
