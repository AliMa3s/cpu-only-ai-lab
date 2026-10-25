"""Fixed-work batching and numerical equivalence tests without pretrained weights."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from cpu_ai_lab.pipeline_speed import (
    batches_by_length, compare_predictions, optimized_evaluate,
    optimized_train_step, paired_statistics, validate_config,
)
from cpu_ai_lab.real_task import evaluate


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([0.01, 0.08, -0.02, -0.04]))
        self.bias = torch.nn.Parameter(torch.tensor([0.0, -0.1, 0.0, 0.2]))

    def forward(self, input_ids, attention_mask, position_ids, **kwargs):
        assert kwargs["logits_to_keep"] == 1
        assert kwargs["use_cache"] is False
        value = (input_ids * attention_mask).sum(-1) / attention_mask.sum(-1)
        logits = value[:, None] * self.weight[None, :] + self.bias[None, :]
        return SimpleNamespace(logits=logits[:, None, :])


def examples():
    return [{"id": index, "text_sha256": f"hash-{index}", "input_ids": list(range(1, length + 1)),
             "label": index % 2, "truncated": False}
            for index, length in enumerate((2, 9, 3, 7, 1, 8, 2))]


def test_length_batches_keep_every_index_once_and_stable_ties():
    rows = examples()
    batches = batches_by_length(rows, 3, True)
    flattened = [i for batch in batches for i in batch]
    assert sorted(flattened) == list(range(len(rows)))
    assert flattened == [4, 0, 6, 2, 3, 5, 1]
    assert batches_by_length(rows, 3, False) == [[0, 1, 2], [3, 4, 5], [6]]
    with pytest.raises(ValueError):
        batches_by_length([], 3, True)


def test_sorted_evaluation_restores_record_order_and_preserves_scores():
    model, rows = TinyClassifier(), examples()
    original_rows = copy.deepcopy(rows)
    baseline = evaluate(model, rows, 0, [1, 3], 3)
    candidate = optimized_evaluate(model, rows, 0, [1, 3], 3, sort_by_length=True)
    assert rows == original_rows
    assert [record["id"] for record in candidate["records"]] == list(range(len(rows)))
    assert compare_predictions(baseline, candidate)["accepted"]
    assert candidate["binary_label_loss"] == pytest.approx(baseline["binary_label_loss"], abs=1e-6)
    plain = optimized_evaluate(model, rows, 0, [1, 3], 3, sort_by_length=False)
    assert candidate["execution"]["input_tokens"] == plain["execution"]["input_tokens"]
    assert candidate["execution"]["padding_tokens"] < plain["execution"]["padding_tokens"]
    assert candidate["execution"]["batches"] == 3


@pytest.mark.parametrize("microbatch", [1, 2, 3, 4])
def test_microbatching_keeps_mean_loss_adam_update_and_one_step(microbatch):
    rows = examples()[:4]
    reference, candidate = TinyClassifier(), TinyClassifier()
    baseline_optimizer = torch.optim.AdamW(reference.parameters(), lr=0.001, weight_decay=0)
    candidate_optimizer = torch.optim.AdamW(candidate.parameters(), lr=0.001, weight_decay=0)
    baseline = optimized_train_step(reference, baseline_optimizer, rows, 0, [1, 3], microbatch_size=4)
    result = optimized_train_step(candidate, candidate_optimizer, rows, 0, [1, 3], microbatch_size=microbatch)
    assert result["binary_label_loss"] == pytest.approx(baseline["binary_label_loss"], abs=1e-6)
    assert result["input_tokens"] == baseline["input_tokens"]
    assert result["global_batch_size"] == 4
    for first, second in zip(reference.parameters(), candidate.parameters()):
        assert torch.allclose(first, second, atol=1e-7, rtol=1e-5)
    for first, second in zip(baseline_optimizer.state.values(), candidate_optimizer.state.values()):
        assert float(first["step"]) == float(second["step"]) == 1
        assert torch.allclose(first["exp_avg"], second["exp_avg"], atol=1e-7)
        assert torch.allclose(first["exp_avg_sq"], second["exp_avg_sq"], atol=1e-8)


def test_gate_checks_content_identity_and_score_drift():
    baseline = optimized_evaluate(TinyClassifier(), examples(), 0, [1, 3], 3, capture_logits=True)
    altered = copy.deepcopy(baseline)
    altered["records"][0]["text_sha256"] = "changed"
    with pytest.raises(ValueError, match="changed"):
        compare_predictions(baseline, altered)
    altered = copy.deepcopy(baseline)
    altered["records"][0]["binary_label_nll"] += 0.01
    assert not compare_predictions(baseline, altered)["accepted"]
    altered = copy.deepcopy(baseline)
    altered["records"][0]["label_logits"][0] += 0.01
    assert not compare_predictions(baseline, altered)["accepted"]


def test_pair_summary_preserves_failed_quality_and_every_repeat():
    rows = [{"baseline_seconds": 2, "candidate_seconds": 1, "quality": {"accepted": True}},
            {"baseline_seconds": 2, "candidate_seconds": 4, "quality": {"accepted": False}},
            {"baseline_seconds": 2, "candidate_seconds": 1, "quality": {"accepted": True}}]
    summary = paired_statistics(rows)
    assert summary["pairs"] == 3
    assert summary["median_paired_speedup"] == 2
    assert summary["aggregate_speedup"] == 1
    assert summary["faster_pairs"] == 2
    assert not summary["all_quality_checks_pass"]


def test_config_rejects_inadequate_confirmation_or_changed_global_budget():
    cfg = json.loads(Path("configs/pipeline-speed.json").read_text())
    assert validate_config(cfg) == cfg
    for change in ({"confirmation_repeats": 2}, {"microbatch_candidates": [4, 8]},
                   {"confirmation_examples": 12}, {"logit_tolerance": float("inf")}):
        with pytest.raises(ValueError):
            validate_config({**cfg, **change})
