"""Select a CPU execution setting from predeclared TRAIN-only measurements."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics


CANDIDATES = ((4, 4), (8, 4), (12, 4), (4, 2))
SOFTWARE = ("calibrate_reader_training.py", "reader_training.py", "reader_model.py", "reader_run.py")
DATA_ACCESS = "Only train.jsonl and provenance/manifest metadata are read. No validation/test row files, predictions or metrics are read."


def digest(payload):
    return hashlib.sha256(payload).hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def payload(value):
    return (json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def numeric(value, *, positive=False):
    return type(value) in (int, float) and math.isfinite(value) and (value > 0 if positive else value >= 0)


def hash_value(value):
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def verify_plan(plan):
    require(plan.get("format") == "cpu-reader-training-calibration-plan-v1", "Unsupported calibration plan.")
    require(plan.get("candidates") == [{"threads": threads, "microbatch_size": size} for threads, size in CANDIDATES],
            "The four predeclared candidates or their tie order changed.")
    fixed = {"training_batches": 8, "warmup_updates": 1, "measured_updates": 7, "global_batch_size": 8,
             "followup_training_seconds_limit": 1800, "followup_peak_rss_limit_gib": 8}
    require(all(type(plan.get(key)) is int and plan[key] == value for key, value in fixed.items()),
            "Predeclared batch, timing or RSS limits changed.")
    require("No validation/test" in plan.get("selection", "") and "median" in plan["selection"],
            "Plan must declare training-time-only selection.")


def current_software():
    scripts = Path(__file__).parent
    source = scripts.parent / "src" / "cpu_ai_lab"
    return {name: digest((scripts / name if name.endswith("calibrate_reader_training.py") else source / name)
                         .read_text(encoding="utf-8").encode("utf-8")) for name in SOFTWARE}


def verify_candidate(result, original, plan, threads, microbatch, software):
    require(result.get("format") == "cpu-reader-training-calibration-v1", "Unsupported candidate format.")
    require(result.get("threads") == threads and result.get("microbatch_size") == microbatch,
            "Candidate does not match its declared filename/settings.")
    expected_config = {**original, "training_threads": threads, "evaluation_threads": threads}
    require(result.get("config") == expected_config, "Candidate changed a non-thread experiment setting.")
    require(original["training_batch_size"] == 8 and original["peak_rss_limit_gib"] == 8,
            "Original global batch or RSS budget differs from the plan.")
    require(result.get("declared_candidates") == plan["candidates"], "Candidate plan differs.")
    require(result.get("data_access") == DATA_ACCESS, "Candidate does not declare TRAIN-only row access.")
    require(result.get("software_sha256") == software, "Candidate implementation fingerprints differ.")
    require(result.get("reader_weights_sha256") == original["reader_weights_sha256"], "Initializer differs.")
    for key in ("train_data_sha256", "split_manifest_sha256", "ordered_calibration_ids_sha256"):
        require(hash_value(result.get(key)), f"Invalid source binding: {key}")
    planned = result.get("planned_global_batch_ids")
    require(isinstance(planned, list) and len(planned) == 8 and
            all(isinstance(batch, list) and len(batch) == 8 and all(isinstance(value, str) and value for value in batch)
                for batch in planned), "Eight complete planned global batches are required.")
    identifiers = [value for batch in planned for value in batch]
    require(len(set(identifiers)) == 64, "Calibration scheduled duplicate examples.")
    require(digest(json.dumps(planned, separators=(",", ":")).encode()) == result["ordered_calibration_ids_sha256"],
            "Calibration ordered-ID checksum mismatch.")
    steps = result.get("steps")
    require(isinstance(steps, list) and len(steps) <= 8, "Invalid measured update list.")
    require(result.get("completed_updates") == len(steps) and result.get("planned_updates") == 8 and
            result.get("warmup_updates") == 1 and result.get("global_batch_size") == 8,
            "Candidate update counts disagree with raw steps.")
    expected_sizes = [min(microbatch, 8 - offset) for offset in range(0, 8, microbatch)]
    for step, row in enumerate(steps, 1):
        require(row.get("step") == step and row.get("warmup") is (step == 1), "Warmup or update order changed.")
        require(row.get("global_batch_ids") == planned[step - 1], "Measured global batch differs from its schedule.")
        ordered = row.get("microbatch_order_ids", [])
        require(isinstance(ordered, list) and len(ordered) == 8 and sorted(ordered) == sorted(planned[step - 1]),
                "Microbatching dropped or repeated scheduled examples.")
        require(row.get("examples") == 8 and row.get("microbatch_sizes") == expected_sizes and
                row.get("microbatch_count") == len(expected_sizes), "Microbatch size/count differs from its candidate.")
        width = row.get("global_padded_width")
        require(type(width) is int and 0 < width <= original["reader_max_length"], "Invalid global padded width.")
        require(type(row.get("input_tokens")) is int and 0 < row["input_tokens"] <= 8 * width and
                row.get("padded_tokens") == 8 * width, "Global padded work was changed.")
        require(numeric(row.get("seconds"), positive=True), "Step durations must be finite and positive.")
        expected_lr = original["learning_rate"] * (step / original["warmup_steps"] if step <= original["warmup_steps"] else
                      (original["optimizer_steps"] - step + 1) / (original["optimizer_steps"] - original["warmup_steps"]))
        require(numeric(row.get("learning_rate")) and math.isclose(row["learning_rate"], expected_lr, rel_tol=1e-12, abs_tol=1e-15),
                "The scheduled learning rate changed.")
    timed = [row["seconds"] for row in steps[1:]]
    median = statistics.median(timed) if timed else None
    require(result.get("timed_updates") == len(timed), "Timed update count disagrees.")
    supplied_median = result.get("median_timed_step_seconds")
    require((median is None and supplied_median is None) or
            (median is not None and numeric(supplied_median, positive=True) and math.isclose(supplied_median, median, rel_tol=1e-12, abs_tol=1e-12)),
            "Median does not reproduce the raw non-warmup steps.")
    require(result.get("input_tokens") == sum(row["input_tokens"] for row in steps) and
            result.get("padded_tokens") == sum(row["padded_tokens"] for row in steps), "Reported token totals disagree.")
    peak = result.get("sampled_peak_process_rss_bytes")
    require(type(peak) is int and peak > 0, "Peak RSS must be positive sampled bytes.")
    require(numeric(result.get("sampled_peak_process_rss_gib"), positive=True) and
            math.isclose(result["sampled_peak_process_rss_gib"], peak / 1024**3, rel_tol=1e-12), "RSS units disagree.")
    complete, under = len(steps) == 8, peak < plan["followup_peak_rss_limit_gib"] * 1024**3
    require(result.get("completed") is complete and result.get("rss_below_budget") is under and
            result.get("eligible_for_selection") is (complete and under), "Claimed eligibility differs from raw steps/RSS.")
    require(result.get("stop_reason") == ("eight_batches_complete" if complete else "rss_budget_before_next_update"),
            "Candidate stop reason disagrees with completion.")
    cpu = result.get("cpu", {})
    require(cpu.get("cuda_build", "missing") is None and cpu.get("cuda_available") is False and
            cpu.get("parameter_devices") == ["cpu"] and cpu.get("optimizer_devices") == (["cpu"] if steps else []),
            "Candidate lacks CPU-only model/optimizer evidence.")
    return {"threads": threads, "microbatch_size": microbatch, "eligible": complete and under,
            "completed_updates": len(steps), "median_timed_step_seconds": median,
            "sampled_peak_process_rss_bytes": peak, "sampled_peak_process_rss_gib": peak / 1024**3,
            "input_tokens": result["input_tokens"], "padded_tokens": result["padded_tokens"]}


def select(calibrations, plan_path, config_path, output, optimized_config):
    plan, original = read(plan_path), read(config_path)
    verify_plan(plan)
    software = current_software()
    results, summaries, source_hashes = [], [], {}
    for threads, microbatch in CANDIDATES:
        name = f"threads-{threads}-micro-{microbatch}.json"
        source = (calibrations / name).read_bytes()
        result = json.loads(source)
        summaries.append(verify_candidate(result, original, plan, threads, microbatch, software))
        results.append(result)
        source_hashes[name] = digest(source)
    first = results[0]
    for result in results[1:]:
        for key in ("train_data_sha256", "split_manifest_sha256", "planned_global_batch_ids", "ordered_calibration_ids_sha256", "software_sha256"):
            require(result[key] == first[key], f"Candidates used different source/work binding: {key}")
    # Partial RSS-stopped candidates remain visible and ineligible. Every
    # measured update must match work for that same update in every other arm.
    for index in range(8):
        measured = [result["steps"][index] for result in results if len(result["steps"]) > index]
        if measured:
            for row in measured[1:]:
                for key in ("input_tokens", "padded_tokens", "global_padded_width", "microbatch_order_ids"):
                    require(row[key] == measured[0][key], f"Candidates changed measured global work: {key}")
    eligible = [index for index, row in enumerate(summaries) if row["eligible"]]
    require(eligible, "No candidate completed all eight updates strictly below the RSS budget.")
    chosen = min(eligible, key=lambda index: (summaries[index]["median_timed_step_seconds"], index))
    winner = summaries[chosen]
    optimized = {**original, "training_threads": winner["threads"], "training_microbatch_size": winner["microbatch_size"],
                 "training_seconds_limit": plan["followup_training_seconds_limit"]}
    config_bytes = payload(optimized)
    allowed = ("training_threads", "training_microbatch_size", "training_seconds_limit")
    require({key: value for key, value in optimized.items() if key not in allowed} ==
            {key: value for key, value in original.items() if key not in allowed}, "Unexpected optimized configuration change.")
    selection = {"format": "cpu-reader-training-calibration-selection-v1", "selected": winner,
                 "candidates": summaries, "source_sha256": source_hashes,
                 "plan_sha256": digest(Path(plan_path).read_bytes()), "original_config_sha256": digest(Path(config_path).read_bytes()),
                 "optimized_config_sha256": digest(config_bytes), "selector_software_sha256": digest(Path(__file__).read_text(encoding="utf-8").encode()),
                 "calibration_software_sha256": software, "train_data_sha256": first["train_data_sha256"],
                 "split_manifest_sha256": first["split_manifest_sha256"], "ordered_calibration_ids_sha256": first["ordered_calibration_ids_sha256"],
                 "verified": {"all_four_predeclared_candidates": True, "identical_eight_planned_global_batches": True,
                              "identical_input_and_padded_work_per_measured_update": True, "global_mean_objective_and_padding_unchanged": True,
                              "cpu_only_evidence": True, "medians_and_eligibility_recomputed": True,
                              "same_frozen_calibration_implementation": True},
                 "changed_settings": {key: {"before": original.get(key), "after": optimized[key]} for key in allowed},
                 "selection_rule": "Lowest recomputed median of updates 2–8 among complete candidates strictly below 8 GiB sampled peak RSS; earlier predeclared candidate breaks exact ties.",
                 "validation_or_test_used_for_selection": False, "training_loss_used_for_selection": False,
                 "limitations": ["A seven-step median is a bounded timing sample, not a full-run speedup estimate.",
                                 "The selected run retains the RSS guard; early calibration cannot guarantee later peak memory.",
                                 "Source hashes and row-access declarations support the training-only protocol; this selector does not reconstruct historical process execution."]}
    selection_bytes = payload(selection)
    require(Path(output).resolve() != Path(optimized_config).resolve(), "Selection and optimized config need distinct paths.")
    # Check both destinations before writing either artifact.
    for path, content in ((Path(output), selection_bytes), (Path(optimized_config), config_bytes)):
        if path.exists() and path.read_bytes() != content:
            raise FileExistsError("Existing different selection/configuration bytes are preserved.")
    for path, content in ((Path(output), selection_bytes), (Path(optimized_config), config_bytes)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibrations", type=Path, default=Path("results/reader-calibration"))
    parser.add_argument("--plan", type=Path, default=Path("configs/reader-training-calibration.json"))
    parser.add_argument("--config", type=Path, default=Path("configs/reader.json"))
    parser.add_argument("--output", type=Path, default=Path("results/reader-calibration/selection.json"))
    parser.add_argument("--optimized-config", type=Path, default=Path("configs/reader-optimized.json"))
    args = parser.parse_args()
    result = select(args.calibrations, args.plan, args.config, args.output, args.optimized_config)
    print(json.dumps({"selected": result["selected"], "changed_settings": result["changed_settings"]}, indent=2))


if __name__ == "__main__":
    main()
