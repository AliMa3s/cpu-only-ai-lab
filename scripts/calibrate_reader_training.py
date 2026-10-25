"""One fresh-process TRAIN-only reader timing candidate; never inspect outcomes.

Run each declared thread/microbatch candidate in its own process. The first of
eight fixed training batches warms up; the remaining seven supply step timing.
The parent chooses the fastest completed candidate below the fixed RSS budget.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch

from cpu_ai_lab import reader_data as data
from cpu_ai_lab import reader_model as reader
from cpu_ai_lab import reader_run as runner
from cpu_ai_lab.reader_training import backward_batch
from cpu_ai_lab.training import PeakRSS, write_json


CANDIDATES = ((4, 4), (8, 4), (12, 4), (4, 2))
BATCHES = 8
WARMUP_BATCHES = 1


def read_training(prepared, config):
    """Open train.jsonl and metadata only; no validation/test row-file reads."""
    data.validate_config(config)
    provenance = runner.read(prepared / "data_provenance.json")
    if provenance["data_config"] != {key: config[key] for key in data.DATA_KEYS}:
        raise ValueError("Calibration data configuration differs from prepared TRAIN.")
    controls = provenance["historical_exclusion_controls"]
    if provenance["prior_manifest_sha256"] != [entry["sha256"] for entry in controls["prior_manifests"]]:
        raise ValueError("Historical manifest bindings disagree.")
    manifest_bytes = (prepared / "split_manifest.json").read_bytes()
    if data.sha256(manifest_bytes) != provenance["split_manifest_sha256"]:
        raise ValueError("Calibration split manifest checksum mismatch.")
    payload = (prepared / "data" / "train.jsonl").read_bytes()
    if data.sha256(payload) != provenance["local_data_sha256"]["train"]:
        raise ValueError("Calibration TRAIN checksum mismatch.")
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    if len(rows) != config["train_size"] or len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Calibration TRAIN count/identities differ.")
    if data.public_manifest({"train": rows})["train"] != json.loads(manifest_bytes)["train"]:
        raise ValueError("Calibration TRAIN rows differ from their manifest.")
    data.check_historical_exclusions({"train": rows}, controls)
    return rows, provenance


def calibrate(config, prepared, threads, microbatch_size, output):
    if (threads, microbatch_size) not in CANDIDATES:
        raise ValueError("Use one of the four predeclared thread/microbatch candidates.")
    if output.exists():
        raise FileExistsError("Existing calibration evidence is preserved.")
    config = dict(config)
    if config["training_batch_size"] != 8:
        raise ValueError("Calibration preserves the declared global batch of eight.")
    config["training_threads"] = threads
    config["evaluation_threads"] = threads
    runner.setup(config)
    started = time.perf_counter()
    rows, provenance = read_training(prepared, config)
    batches = runner.schedule(rows, config)[:BATCHES]
    if len(batches) != BATCHES:
        raise ValueError("Eight complete TRAIN batches are required for calibration.")
    first_ids = [[rows[index]["id"] for index in batch] for batch in batches]
    completed, stop_reason = [], "eight_batches_complete"
    with PeakRSS() as memory:
        model, tokenizer = reader.load_initial(runner.initializer_path(config), seed=config["seed"])
        # Encode only the predetermined 64 selected TRAIN rows, outside timing.
        selected = [index for batch in batches for index in batch]
        local_index = {index: number for number, index in enumerate(selected)}
        features, positions = runner.encoded_rows([rows[index] for index in selected], tokenizer, config)
        model.requires_grad_(True)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                     weight_decay=config["weight_decay"], foreach=False)
        for step, batch_indices in enumerate(batches, 1):
            if memory.peak_bytes >= config["peak_rss_limit_gib"] * 1024**3:
                stop_reason = "rss_budget_before_next_update"
                break
            tick = time.perf_counter()
            rate = config["learning_rate"] * (step / config["warmup_steps"] if step <= config["warmup_steps"] else
                   (config["optimizer_steps"] - step + 1) / (config["optimizer_steps"] - config["warmup_steps"]))
            for group in optimizer.param_groups:
                group["lr"] = rate
            optimizer.zero_grad(set_to_none=True)
            result = backward_batch(model, features, positions, [local_index[index] for index in batch_indices],
                                    tokenizer.pad_token_id, microbatch_size)
            gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True)
            optimizer.step()
            seconds = time.perf_counter() - tick
            completed.append({"step": step, "warmup": step <= WARMUP_BATCHES, "seconds": seconds,
                              "learning_rate": rate, "gradient_norm_before_clipping": float(gradient),
                              "global_batch_ids": first_ids[step - 1],
                              "microbatch_order_ids": [rows[selected[index]]["id"] for index in result["ordered_indices"]],
                              **{key: value for key, value in result.items() if key != "ordered_indices"}})
            print(json.dumps({"stage": "calibrate_reader_training", "threads": threads,
                              "microbatch_size": microbatch_size, "step": step, "seconds": seconds,
                              "peak_rss_gib": memory.peak_bytes / 1024**3}), flush=True)
        reader.assert_cpu(model)
        optimizer_devices = sorted({str(value.device) for state in optimizer.state.values() for value in state.values()
                                    if isinstance(value, torch.Tensor)})
        elapsed = time.perf_counter() - started
    timed = [row["seconds"] for row in completed if not row["warmup"]]
    complete = len(completed) == BATCHES
    under_budget = memory.peak_bytes < config["peak_rss_limit_gib"] * 1024**3
    result = {"format": "cpu-reader-training-calibration-v1", "created_utc": datetime.now(timezone.utc).isoformat(),
              "config": config, "threads": threads, "microbatch_size": microbatch_size,
              "declared_candidates": [{"threads": first, "microbatch_size": second} for first, second in CANDIDATES],
              "completed": complete, "eligible_for_selection": complete and under_budget,
              "stop_reason": stop_reason, "completed_updates": len(completed), "planned_updates": BATCHES,
              "warmup_updates": WARMUP_BATCHES, "timed_updates": len(timed),
              "median_timed_step_seconds": statistics.median(timed) if timed else None,
              "full_calibration_seconds": elapsed,
              "sampled_peak_process_rss_bytes": memory.peak_bytes,
              "sampled_peak_process_rss_gib": memory.peak_bytes / 1024**3,
              "rss_below_budget": under_budget, "global_batch_size": 8,
              "input_tokens": sum(row["input_tokens"] for row in completed),
              "padded_tokens": sum(row["padded_tokens"] for row in completed),
              "planned_global_batch_ids": first_ids, "steps": completed,
              "train_data_sha256": provenance["local_data_sha256"]["train"],
              "split_manifest_sha256": provenance["split_manifest_sha256"],
              "ordered_calibration_ids_sha256": hashlib.sha256(json.dumps(first_ids, separators=(",", ":")).encode()).hexdigest(),
              "reader_weights_sha256": config["reader_weights_sha256"],
              "software_sha256": {name: runner.code_digest(path) for name, path in (
                  ("calibrate_reader_training.py", Path(__file__)),
                  ("reader_training.py", Path(runner.__file__).with_name("reader_training.py")),
                  ("reader_model.py", Path(runner.__file__).with_name("reader_model.py")),
                  ("reader_run.py", Path(runner.__file__)))},
              "cpu": {"cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                      "parameter_devices": sorted({str(value.device) for value in model.parameters()}),
                      "optimizer_devices": optimizer_devices},
              "selection_rule": "Choose the lowest median of seven timed steps among completed candidates strictly below the fixed RSS budget; no quality metrics or validation/test outcomes enter the choice.",
              "timing_scope": "Per step: scheduled LR, zero_grad, length-sorted microbatch collation/forward/backward with the original GLOBAL batch padded width, clipping, and AdamW update. Padded-token work is unchanged. One warmup update is excluded from median; loading/encoding stay in full calibration time.",
              "data_access": "Only train.jsonl and provenance/manifest metadata are read. No validation/test row files, predictions or metrics are read.",
              "limitations": ["Eight early training batches are a bounded execution calibration, not a guarantee of later peak memory or time.",
                              "Microbatching preserves the global example-mean objective; dropout draws and floating-point reduction order can alter the optimization trajectory.",
                              "RSS is sampled full process resident memory. Each candidate must run in a separate fresh process."]}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return {key: result[key] for key in ("threads", "microbatch_size", "completed_updates", "eligible_for_selection",
                                        "median_timed_step_seconds", "sampled_peak_process_rss_gib")}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/reader.json"))
    parser.add_argument("--prepared-data", type=Path, default=Path("artifacts/reader-data"))
    parser.add_argument("--threads", type=int, required=True)
    parser.add_argument("--microbatch-size", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(calibrate(runner.read(args.config), args.prepared_data, args.threads, args.microbatch_size, args.output), indent=2))


if __name__ == "__main__":
    main()
