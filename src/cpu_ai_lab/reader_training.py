"""Memory-bounded backward passes for the unchanged mean span-loss objective.

The caller owns gradient zeroing, clipping and the one optimizer step per global
batch. Microbatch order may change dropout draws and floating-point reductions;
identical training trajectories are not promised.
"""
from __future__ import annotations

import math

import torch

from . import reader_model as reader


def backward_batch(model, features, positions, indices, pad_token_id, microbatch_size):
    """Accumulate the global example-mean loss gradient, freeing each graph.

    Sort only the supplied global batch by known encoded lengths. Every
    microbatch contributes len(microbatch)/len(global_batch) times its mean
    start/end loss. Existing gradients are accumulated into, never cleared.

    Every microbatch keeps the GLOBAL batch's padded width. BERT's built-in QA
    cross entropy includes padded output positions in its softmax denominator;
    attention masking alone does not remove them from that loss. Shortening a
    microbatch would therefore change the original objective. This trades lower
    simultaneous activation memory for equal padded-token work, not less work.
    """
    if type(microbatch_size) is not int or microbatch_size < 1:
        raise ValueError("Microbatch size must be a positive integer.")
    if len(features) != len(positions) or not indices:
        raise ValueError("Aligned features/positions and a nonempty global batch are required.")
    if any(type(index) is not int or not 0 <= index < len(features) for index in indices):
        raise ValueError("Global batch indices must refer to existing features.")
    if len(set(indices)) != len(indices):
        raise ValueError("A global batch cannot repeat an example index.")
    reader.assert_cpu(model)
    order = sorted(indices, key=lambda index: (len(features[index]["input_ids"]), index))
    global_width = max(len(features[index]["input_ids"]) for index in order)
    count, mean_loss, padded_tokens, input_tokens = len(order), 0.0, 0, 0
    microbatch_sizes = []
    for offset in range(0, count, microbatch_size):
        group = order[offset:offset + microbatch_size]
        batch = reader.collate([features[index] for index in group], pad_token_id,
                               [positions[index] for index in group])
        padding = global_width - batch["input_ids"].shape[1]
        if padding:
            for key in reader.INPUT_KEYS:
                if key in batch:
                    batch[key] = torch.nn.functional.pad(batch[key], (0, padding),
                                                        value=pad_token_id if key == "input_ids" else 0)
        loss = model(**batch).loss
        if loss.numel() != 1 or not bool(torch.isfinite(loss)):
            raise RuntimeError("A microbatch must return a finite scalar mean span loss.")
        weight = len(group) / count
        scalar = float(loss.detach())
        (loss * weight).backward()
        mean_loss += scalar * weight
        padded_tokens += int(batch["input_ids"].numel())
        input_tokens += sum(len(features[index]["input_ids"]) for index in group)
        microbatch_sizes.append(len(group))
        # Do not retain previous activations while constructing the next batch.
        del loss, batch
    if not math.isfinite(mean_loss):
        raise RuntimeError("Accumulated global mean loss is nonfinite.")
    return {"mean_loss": mean_loss, "input_tokens": input_tokens, "padded_tokens": padded_tokens,
            "global_padded_width": global_width,
            "examples": count, "microbatch_sizes": microbatch_sizes,
            "microbatch_count": len(microbatch_sizes), "ordered_indices": order}
