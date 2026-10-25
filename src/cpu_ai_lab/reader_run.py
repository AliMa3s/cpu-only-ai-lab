"""CPU adaptation and controlled evaluation of a compact extractive QA reader."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import random
import statistics
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import torch
from huggingface_hub import snapshot_download

from . import document_v2 as generator
from . import document_v2_metrics as scoring
from . import reader_data as data
from . import reader_metrics as metrics
from . import reader_model as reader
from .reader_training import backward_batch
from .training import PeakRSS, adapter_snapshot, write_json


SOFTWARE = ("reader_run.py", "reader_data.py", "reader_model.py", "reader_metrics.py", "reader_training.py",
            "document_v2.py", "document_v2_data.py", "document_v2_metrics.py")


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_digest(path):
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def now():
    return datetime.now(timezone.utc).isoformat()


def compact(value):
    return {key: item for key, item in value.items() if key != "records"}


def validate_config(config):
    for key in ("train_size", "validation_size", "test_size", "training_batch_size", "optimizer_steps",
                "reader_eval_batch_size", "generation_batch_size", "generation_tokens", "training_threads", "evaluation_threads",
                "training_seconds_limit", "benchmark_per_answerability", "benchmark_repetitions"):
        if type(config.get(key)) is not int or config[key] <= 0:
            raise ValueError(f"Invalid positive integer: {key}")
    if config["training_batch_size"] % 2 or config["train_size"] % config["training_batch_size"]:
        raise ValueError("Balanced training requires even complete batches.")
    if config["optimizer_steps"] * config["training_batch_size"] != config["train_size"]:
        raise ValueError("This recipe performs exactly one scheduled pass over training examples.")
    microbatch = config.get("training_microbatch_size", config["training_batch_size"])
    if type(microbatch) is not int or not 1 <= microbatch <= config["training_batch_size"]:
        raise ValueError("Training microbatch size must be between one and the global batch size.")
    if not 0 < config["warmup_steps"] < config["optimizer_steps"]:
        raise ValueError("Warmup must be shorter than training.")
    fixed = {"reader_model_id": reader.MODEL_ID, "reader_revision": reader.REVISION,
             "reader_weights_sha256": reader.INITIAL_WEIGHTS_SHA256,
             "minimum_correct_answer_retention": 0.95, "minimum_missing_answer_recall": 0.4,
             "maximum_answerable_f1_loss": 0.02}
    if any(config.get(key) != value for key, value in fixed.items()):
        raise ValueError("Pinned reader or quality criteria differ from the declared experiment.")
    for key in ("learning_rate", "weight_decay", "gradient_clip_norm", "peak_rss_limit_gib"):
        if type(config.get(key)) not in (int, float) or not math.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"Invalid positive finite value: {key}")


def setup(config):
    validate_config(config)
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("A CPU-only PyTorch build is required.")
    torch.set_num_threads(config["evaluation_threads"])
    torch.set_num_interop_threads(1)
    torch.manual_seed(config["seed"])
    random.seed(config["seed"])
    torch.use_deterministic_algorithms(True)


def initializer_path(config, download=False):
    path = Path(snapshot_download(config["reader_model_id"], revision=config["reader_revision"],
                                 local_files_only=not download,
                                 allow_patterns=["model.safetensors", "config.json", "tokenizer.json", "tokenizer_config.json",
                                                 "special_tokens_map.json", "vocab.txt", "README.md"]))
    if digest(path / "model.safetensors") != config["reader_weights_sha256"]:
        raise ValueError("Initializer weights checksum mismatch.")
    return path


def load_generator(config, source):
    model, tokenizer, parameters, evidence = generator.load_source(config, source)
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer, parameters, evidence


def schedule(rows, config):
    rng = random.Random(config["seed"])
    groups = [[index for index, row in enumerate(rows) if row["is_impossible"] == missing] for missing in (False, True)]
    if len(groups[0]) != len(groups[1]):
        raise ValueError("Training answerability must be balanced.")
    for values in groups:
        rng.shuffle(values)
    half = config["training_batch_size"] // 2
    batches = []
    for start in range(0, len(groups[0]), half):
        batch = groups[0][start:start + half] + groups[1][start:start + half]
        rng.shuffle(batch)
        batches.append(batch)
    if len(batches) != config["optimizer_steps"] or sorted(i for batch in batches for i in batch) != list(range(len(rows))):
        raise ValueError("Schedule must visit every training example exactly once.")
    return batches


def encoded_rows(rows, tokenizer, config):
    encoded, positions = [], []
    for row in rows:
        value = reader.encode_example(tokenizer, row["question"], row["context"], config["reader_max_length"])
        position = reader.training_positions(value, row["context"], row["literal_answer_spans"],
                                             row["is_impossible"], config["reader_max_answer_tokens"])
        encoded.append(value)
        positions.append(position)
    return encoded, positions


@torch.inference_mode()
def loss_evaluation(model, tokenizer, rows, config):
    model.eval()
    encoded, positions = encoded_rows(rows, tokenizer, config)
    order = sorted(range(len(rows)), key=lambda i: (len(encoded[i]["input_ids"]), i))
    total = 0.0
    for start in range(0, len(order), config["reader_eval_batch_size"]):
        indices = order[start:start + config["reader_eval_batch_size"]]
        batch = reader.collate([encoded[i] for i in indices], tokenizer.pad_token_id, [positions[i] for i in indices])
        value = float(model(**batch).loss)
        if not math.isfinite(value):
            raise RuntimeError("Nonfinite evaluation loss.")
        total += value * len(indices)
    return {"mean_span_loss": total / len(rows), "examples": len(rows),
            "definition": "Mean of start/end cross entropy per example, including native null CLS labels; no answer-aware cropping."}


def reader_evaluation(model, tokenizer, rows, config, label):
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["reader_tokens"], i))
    output = [None] * len(rows)
    tick = time.perf_counter()
    size = config["reader_eval_batch_size"]
    for start in range(0, len(order), size):
        indices = order[start:start + size]
        values = reader.predict_batch(model, tokenizer, [rows[i] for i in indices],
                                      config["reader_max_length"], config["reader_max_answer_tokens"])
        for i, value in zip(indices, values, strict=True):
            output[i] = {"id": rows[i]["id"], **value}
        if start % (size * 4) == 0 or start + size >= len(rows):
            print(json.dumps({"stage": label, "processed": min(start + size, len(rows)), "total": len(rows)}), flush=True)
    return output, time.perf_counter() - tick


def choose_checkpoint(selections):
    """Two predeclared checkpoints only; test metrics are never an input."""
    passing = [name for name in ("initial", "final") if selections[name]["accepted"]]
    if passing:
        selected = max(passing, key=lambda name: (selections[name]["selected"]["answerable_token_f1"],
                                                  selections[name]["selected"]["correct_abstention_recall"], name == "initial"))
        return {"kind": "reader", "checkpoint": selected, "threshold": selections[selected]["selected_threshold"],
                "diagnostic_checkpoint": selected, "diagnostic_threshold": selections[selected]["selected_threshold"]}
    diagnostic = max(("initial", "final"), key=lambda name: (
        selections[name]["diagnostic_candidate"]["gate"]["retention_and_f1_pass"],
        selections[name]["diagnostic_candidate"]["gate"]["exact_correct_retention"] or 0.0,
        selections[name]["diagnostic_candidate"]["metrics"]["answerable_token_f1"],
        selections[name]["diagnostic_candidate"]["metrics"]["correct_abstention_recall"], name == "initial"))
    return {"kind": "generator", "checkpoint": None, "threshold": None,
            "diagnostic_checkpoint": diagnostic, "diagnostic_threshold": selections[diagnostic]["diagnostic_threshold"]}


def reader_metric(rows, local, threshold, generator_predictions):
    return metrics.evaluate_reader(rows, [value["span_prediction"] for value in local],
                                   [value["null_minus_span_score"] for value in local], threshold, generator_predictions)


def public_decoded(local):
    return [{**{key: value for key, value in row.items() if key != "span_prediction"},
             "span_prediction_sha256": scoring.sha256(row["span_prediction"])} for row in local]


def run(config, output, prepared, source):
    validate_config(config)
    if output.exists():
        raise FileExistsError("Use a new run directory; existing experiments are preserved.")
    output.mkdir(parents=True)
    write_json(output / "resolved_config.json", config)
    write_json(output / "run_status.json", {"status": "running", "started_utc": now()})
    setup(config)
    started = time.perf_counter()
    try:
        with PeakRSS() as memory:
            splits, provenance = data.load_prepared(prepared, config)
            write_json(output / "data_provenance.json", provenance)
            (output / "split_manifest.json").write_bytes((prepared / "split_manifest.json").read_bytes())
            initial_path = initializer_path(config)
            source_metrics = read(source / "metrics.json")
            if provenance["prior_manifest_sha256"][0] != source_metrics["dataset"]["split_manifest_sha256"]:
                raise ValueError("Historical exclusions do not bind to the generator source.")
            protocol = {"format": "cpu-reader-protocol-v1", "created_utc": now(), "config": config,
                        "reader_source": {"id": reader.MODEL_ID, "revision": reader.REVISION,
                                          "weights_sha256": reader.INITIAL_WEIGHTS_SHA256, "license": reader.MODEL_LICENSE,
                                          "previous_training": "Publisher fine-tuned on SQuAD2 using a GPU; current experiment adapts and infers on CPU only.",
                                          "card": "https://huggingface.co/deepset/minilm-uncased-squad2"},
                        "generator_source_metrics_sha256": digest(source / "metrics.json"),
                        "prompt_instruction": generator.data.INSTRUCTION,
                        "software_sha256": {name: code_digest(Path(__file__).with_name(name)) for name in SOFTWARE},
                        "software_hash_encoding": "UTF-8, universal newlines normalized to LF",
                        "selection": "Initial and final checkpoints each get a validation-only fixed-grid threshold. Among passing checkpoints prefer higher answerable F1, then missing recall, then initial on ties. Otherwise generator fallback.",
                        "diagnostic": "On failure rank the two diagnostic candidates by quality-constraint pass, paired retention, answerable F1, missing recall, initial on ties; never use test outcomes.",
                        "test_policy": "No test generation or reader forward pass before policy.json is serialized.",
                        "training": "Full-parameter CPU AdamW, one balanced scheduled pass, fixed warmup/linear decay, gradient clipping. Microbatches preserve global padded width and example-mean loss; zero/clip/step occur once per global batch. Dropout draws and floating-point order can differ. Time/RSS guards may stop early; actual updates are reported.",
                        "interpretation": "Different architectures and prior training histories; gains cannot be attributed to architecture or this CPU adaptation alone."}
            write_json(output / "protocol.json", protocol)
            stages = {}
            tick = time.perf_counter()
            gen, gen_tokenizer, gen_parameters, gen_source = load_generator(config, source)
            gen_initial = generator.state_hash(adapter_snapshot(gen))
            gen_validation, gen_val_local = generator.generation_evaluation(gen, gen_tokenizer, splits["validation"],
                                                                           config["generation_batch_size"], config["generation_tokens"])
            if generator.state_hash(adapter_snapshot(gen)) != gen_initial:
                raise RuntimeError("Frozen generator changed during inference.")
            stages["generator_validation_including_load"] = time.perf_counter() - tick
            write_json(output / "generator-validation-local.json", gen_val_local)
            write_json(output / "generator-validation.json", gen_validation)
            gen_val_predictions = [value["prediction"] for value in gen_val_local]
            del gen, gen_tokenizer
            gc.collect()
            model, tokenizer = reader.load_initial(initial_path, seed=config["seed"])
            initial_hash = generator.state_hash(model.state_dict())
            initial_manifest = reader.save_reader(output / "initial_reader", model, tokenizer, {"checkpoint": "initial", "source": protocol["reader_source"]})
            initial_validation, stages["initial_reader_validation"] = reader_evaluation(model, tokenizer, splits["validation"], config, "initial_reader_validation")
            initial_loss = loss_evaluation(model, tokenizer, splits["validation"], config)
            write_json(output / "initial-validation-local.json", initial_validation)
            features, positions = encoded_rows(splits["train"], tokenizer, config)
            batches = schedule(splits["train"], config)
            write_json(output / "sample_schedule.json", [[splits["train"][i]["id"] for i in batch] for batch in batches])
            model.requires_grad_(True)
            trainable_parameters = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
            parameter_devices = sorted({str(parameter.device) for parameter in model.parameters()})
            torch.set_num_threads(config["training_threads"])
            optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"], foreach=False)
            curve, seen, stop_reason = [], set(), "scheduled_pass_complete"
            training_tick = time.perf_counter()
            model.train()
            for step, indices in enumerate(batches, 1):
                if step > 1 and time.perf_counter() - training_tick >= config["training_seconds_limit"]:
                    stop_reason = "time_budget_after_completed_update"
                    break
                if memory.peak_bytes > config["peak_rss_limit_gib"] * 1024**3:
                    stop_reason = "rss_budget_before_next_update"
                    break
                rate = config["learning_rate"] * (step / config["warmup_steps"] if step <= config["warmup_steps"] else
                       (config["optimizer_steps"] - step + 1) / (config["optimizer_steps"] - config["warmup_steps"]))
                for group in optimizer.param_groups:
                    group["lr"] = rate
                optimizer.zero_grad(set_to_none=True)
                tick = time.perf_counter()
                backward = backward_batch(model, features, positions, indices, tokenizer.pad_token_id,
                                          config.get("training_microbatch_size", config["training_batch_size"]))
                gradient = torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip_norm"], error_if_nonfinite=True)
                optimizer.step()
                seen.update(indices)
                curve.append({"step": step, "loss": backward["mean_loss"], "learning_rate": rate,
                              "gradient_norm_before_clipping": float(gradient), "compute_seconds": time.perf_counter() - tick,
                              "input_tokens": backward["input_tokens"], "padded_tokens": backward["padded_tokens"],
                              "microbatch_sizes": backward["microbatch_sizes"], "microbatch_count": backward["microbatch_count"],
                              "global_padded_width": backward["global_padded_width"]})
                if step % 16 == 0 or step == 1 or step == len(batches):
                    print(json.dumps({"stage": "train_reader", "step": step, "planned": len(batches), "loss": curve[-1]["loss"],
                                      "seconds": time.perf_counter() - training_tick}), flush=True)
            stages["reader_training"] = time.perf_counter() - training_tick
            optimizer_devices = sorted({str(value.device) for state in optimizer.state.values() for value in state.values() if isinstance(value, torch.Tensor)})
            del optimizer, features, positions
            gc.collect()
            model.eval()
            torch.set_num_threads(config["evaluation_threads"])
            final_hash = generator.state_hash(model.state_dict())
            final_manifest = reader.save_reader(output / "final_reader", model, tokenizer,
                                                 {"checkpoint": "final", "updates": len(curve), "source": protocol["reader_source"]})
            final_validation, stages["final_reader_validation"] = reader_evaluation(model, tokenizer, splits["validation"], config, "final_reader_validation")
            final_loss = loss_evaluation(model, tokenizer, splits["validation"], config)
            write_json(output / "final-validation-local.json", final_validation)
            write_json(output / "training_curve.json", curve)
            selections = {name: metrics.select_threshold(splits["validation"], [v["span_prediction"] for v in local],
                                                         [v["null_minus_span_score"] for v in local], gen_val_predictions)
                          for name, local in (("initial", initial_validation), ("final", final_validation))}
            choice = choose_checkpoint(selections)
            selection = {"format": "cpu-reader-selection-v1", "completed_utc": now(), "uses_test": False,
                         "checkpoints": selections, "choice": choice}
            write_json(output / "selection.json", selection)
            policy = {"format": "cpu-reader-policy-v1", "created_utc": now(), "generation_config": config, **choice,
                      "generator_source": gen_source, "selection_uses_test": False,
                      "selection_sha256": digest(output / "selection.json"), "protocol_sha256": digest(output / "protocol.json"),
                      "reader_manifest_sha256": {name: digest(output / (name + "_reader") / "reader_manifest.json") for name in ("initial", "final")}}
            write_json(output / "policy.json", policy)
            policy_hash = digest(output / "policy.json")
            print(json.dumps({"stage": "policy_frozen_before_test", **choice}), flush=True)
            del model, tokenizer
            gc.collect()
            test_started = now()
            tick = time.perf_counter()
            gen, gen_tokenizer, _, source_again = load_generator(config, source)
            if source_again != gen_source:
                raise ValueError("Generator source changed between validation and test.")
            gen_test, gen_test_local = generator.generation_evaluation(gen, gen_tokenizer, splits["test"],
                                                                     config["generation_batch_size"], config["generation_tokens"])
            stages["generator_test_including_reload"] = time.perf_counter() - tick
            write_json(output / "generator-test-local.json", gen_test_local)
            write_json(output / "generator-test.json", gen_test)
            gen_test_predictions = [value["prediction"] for value in gen_test_local]
            del gen, gen_tokenizer
            gc.collect()
            evaluations, test_local, test_losses = {}, {}, {}
            for name in ("initial", "final"):
                model, tokenizer = reader.load_reader(output / (name + "_reader"))
                local, stages[name + "_reader_test"] = reader_evaluation(model, tokenizer, splits["test"], config, name + "_reader_test")
                test_losses[name] = loss_evaluation(model, tokenizer, splits["test"], config)
                test_local[name] = local
                write_json(output / (name + "-test-local.json"), local)
                evaluations[name] = {
                    "fixed_zero": reader_metric(splits["test"], local, 0.0, gen_test_predictions),
                    "diagnostic": reader_metric(splits["test"], local, selections[name]["diagnostic_threshold"], gen_test_predictions),
                    "threshold_selected": reader_metric(splits["test"], local, selections[name]["selected_threshold"], gen_test_predictions)
                                          if selections[name]["accepted"] else None}
                del model, tokenizer
                gc.collect()
            diagnostic = reader_metric(splits["test"], test_local[choice["diagnostic_checkpoint"]], choice["diagnostic_threshold"], gen_test_predictions)
            selected = evaluations[choice["checkpoint"]]["threshold_selected"] if choice["kind"] == "reader" else gen_test
            for name, local in (("initial-validation", initial_validation), ("final-validation", final_validation),
                                ("initial-test", test_local["initial"]), ("final-test", test_local["final"])):
                write_json(output / (name + "-decoded.json"), public_decoded(local))
            if digest(output / "policy.json") != policy_hash:
                raise RuntimeError("Frozen policy changed during test.")
            elapsed = time.perf_counter() - started
        report = {"format": "cpu-reader-report-v1", "completed_utc": now(), "dataset": provenance,
                  "generator_source": gen_source, "reader_source": protocol["reader_source"],
                  "selection": {**choice, "policy_saved_utc": policy["created_utc"], "test_started_utc": test_started, "test_used_for_selection": False},
                  "training": {"trainable_parameters": trainable_parameters, "completed_updates": len(curve), "planned_updates": len(batches),
                               "unique_examples_seen": len(seen), "presentations": len(curve) * config["training_batch_size"],
                               "stop_reason": stop_reason, "initial_state_sha256": initial_hash, "final_state_sha256": final_hash,
                               "weights_changed": initial_hash != final_hash, "validation_loss_initial": initial_loss, "validation_loss_final": final_loss,
                               "training_tokens": sum(row["input_tokens"] for row in curve), "padded_tokens": sum(row["padded_tokens"] for row in curve)},
                  "cpu": {"torch_version": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                          "reader_parameter_devices": parameter_devices, "optimizer_devices": optimizer_devices, "generator_trainable_parameters": 0},
                  "timing": {"full_pipeline_seconds": elapsed, "stages_seconds": stages, "peak_process_rss_bytes": memory.peak_bytes,
                             "peak_process_rss_gib": memory.peak_bytes / 1024**3,
                             "scope": "Model loads, generation, full reader adaptation, validation, selection, checkpoint saves/reloads and test. Excludes data preparation, interpreter startup and final report serialization."},
                  "validation": {"generator": gen_validation,
                                 "initial_fixed_zero": reader_metric(splits["validation"], initial_validation, 0.0, gen_val_predictions),
                                 "final_fixed_zero": reader_metric(splits["validation"], final_validation, 0.0, gen_val_predictions)},
                  "test": {"generator": gen_test, "initial": evaluations["initial"], "final": evaluations["final"],
                           "diagnostic_candidate": diagnostic, "selected": selected, "span_losses": test_losses},
                  "reader_manifests": {"initial": initial_manifest, "final": final_manifest},
                  "limitations": ["Fresh filtered DuoRC movie plots, not the full corpus or arbitrary documents; human answers require at least one bounded exact literal span for eligibility.",
                                  "Multiple questions per movie are clustered; answerability is balanced, passage lengths are not.",
                                  "Reader and generator have unequal prior supervision. The publisher reader was trained on SQuAD2 on a GPU; only this adaptation/inference stage is CPU-only.",
                                  "Historical public data may have appeared in model pretraining; exact exclusions cannot prove absence of semantic contamination.",
                                  "Only initial and one final reader checkpoint are considered. Fixed thresholds and selection use validation only."]}
        names = ["resolved_config.json", "protocol.json", "selection.json", "policy.json", "data_provenance.json", "split_manifest.json",
                 "sample_schedule.json", "training_curve.json", "generator-validation.json", "generator-test.json"]
        names += [name + "-decoded.json" for name in ("initial-validation", "final-validation", "initial-test", "final-test")]
        report["artifact_sha256"] = {name: digest(output / name) for name in names}
        report["local_prediction_sha256"] = {path.name: digest(path) for path in output.glob("*-local.json")}
        write_json(output / "report.json", report)
        write_json(output / "run_status.json", {"status": "complete", "completed_utc": now()})
        return {"selection": report["selection"], "training": report["training"], "timing": report["timing"],
                "test": {name: compact(report["test"][name]) for name in ("generator", "diagnostic_candidate", "selected")}}
    except BaseException as error:
        write_json(output / "run_status.json", {"status": "failed", "error_type": type(error).__name__})
        raise


def load_policy(directory):
    if read(directory / "run_status.json").get("status") != "complete":
        raise ValueError("Load only a completed reader experiment.")
    policy, report = read(directory / "policy.json"), read(directory / "report.json")
    if policy.get("format") != "cpu-reader-policy-v1" or policy.get("selection_uses_test") is not False:
        raise ValueError("Invalid reader policy contract.")
    for name, expected in report["artifact_sha256"].items():
        if Path(name).name != name or digest(directory / name) != expected:
            raise ValueError("Report artifact checksum mismatch.")
    for name in ("protocol", "selection"):
        if digest(directory / (name + ".json")) != policy[name + "_sha256"]:
            raise ValueError("Policy binding checksum mismatch.")
    protocol, selection = read(directory / "protocol.json"), read(directory / "selection.json")
    choice = choose_checkpoint(selection["checkpoints"])
    if selection.get("uses_test") is not False or choice != selection["choice"] or any(policy[key] != value for key, value in choice.items()):
        raise ValueError("Policy does not reproduce validation selection.")
    if policy["generation_config"] != protocol["config"] or policy["generator_source"] != report["generator_source"]:
        raise ValueError("Policy source/configuration differs.")
    if protocol["prompt_instruction"] != generator.data.INSTRUCTION:
        raise ValueError("Generator prompt changed.")
    validate_config(policy["generation_config"])
    for name in SOFTWARE:
        if protocol["software_sha256"].get(name) != code_digest(Path(__file__).with_name(name)):
            raise ValueError("Implementation differs from frozen protocol.")
    return policy, report


def load_checkpoint(directory, policy, stage):
    if stage not in ("initial", "final"):
        raise ValueError("Unknown reader checkpoint.")
    path = directory / (stage + "_reader")
    if digest(path / "reader_manifest.json") != policy["reader_manifest_sha256"][stage]:
        raise ValueError("Reader checkpoint manifest differs from policy.")
    return reader.load_reader(path)


def fixed_benchmark_rows(rows, per_group):
    counts = {False: 0, True: 0}
    chosen = []
    for row in rows:
        group = row["is_impossible"]
        if counts[group] < per_group:
            chosen.append(row)
            counts[group] += 1
    if any(count != per_group for count in counts.values()):
        raise ValueError("Insufficient fixed benchmark rows.")
    return chosen


def benchmark(directory, prepared, source, mode, output):
    if output.exists():
        raise FileExistsError("Preserving previous benchmark.")
    policy, report = load_policy(directory)
    config = policy["generation_config"]
    setup(config)
    splits, provenance = data.load_prepared(prepared, config)
    if provenance != report["dataset"]:
        raise ValueError("Benchmark dataset differs.")
    rows = fixed_benchmark_rows(splits["test"], config["benchmark_per_answerability"])
    expected = {"generator": {row["id"]: row for row in report["test"]["generator"]["records"]},
                "reader": {row["id"]: row for row in report["test"]["diagnostic_candidate"]["records"]}}
    expected_raw = {row["id"]: row for row in read(directory / (policy["diagnostic_checkpoint"] + "-test-decoded.json"))}
    selected_arms = ("generator", "reader") if mode == "paired" else (mode,)
    tick = time.perf_counter()
    with PeakRSS() as memory:
        models = {}
        if "generator" in selected_arms:
            gen, tok, _, evidence = load_generator(config, source)
            if evidence != policy["generator_source"]:
                raise ValueError("Generator source differs from policy.")
            models["generator"] = (gen, tok)
        if "reader" in selected_arms:
            models["reader"] = load_checkpoint(directory, policy, policy["diagnostic_checkpoint"])
        loading = time.perf_counter() - tick

        def request(row, arm):
            model, tokenizer = models[arm]
            started = time.perf_counter()
            if arm == "generator":
                generated = generator.answer(model, tokenizer, config, row["context"], row["question"])
                answer = generated["answer"]
                raw_hash = generated["prediction_sha256"]
                decoded = None
            else:
                decoded = reader.predict_batch(model, tokenizer, [row], config["reader_max_length"], config["reader_max_answer_tokens"])[0]
                answer = metrics.apply_threshold(decoded["span_prediction"], decoded["null_minus_span_score"], policy["diagnostic_threshold"])
                raw_hash = scoring.sha256(answer)
            seconds = time.perf_counter() - started
            value = {"id": row["id"], "movie_sha256": row["article_sha256"], "is_impossible": row["is_impossible"],
                     "arm": arm, "seconds": seconds, "context_sha256": row["context_sha256"],
                     "question_sha256": row["question_sha256"], "prediction_sha256": raw_hash,
                     "displayed_answer_sha256": scoring.sha256(answer),
                     "matches_saved_batched_output": raw_hash == expected[arm][row["id"]]["prediction_sha256"]}
            if decoded is not None:
                value["span_matches_saved_batch"] = scoring.sha256(decoded["span_prediction"]) == expected_raw[row["id"]]["span_prediction_sha256"]
                value["null_score_difference_from_saved_batch"] = decoded["null_minus_span_score"] - expected_raw[row["id"]]["null_minus_span_score"]
            return value

        warmups = [request(rows[0], arm) for arm in selected_arms]
        records, pairs = [], []
        repetitions = config["benchmark_repetitions"] if mode == "paired" else 1
        for repeat in range(repetitions):
            for index, row in enumerate(rows):
                arms = ("generator", "reader") if (repeat + index) % 2 == 0 else ("reader", "generator")
                if mode != "paired":
                    arms = (mode,)
                current = {}
                for arm in arms:
                    current[arm] = {**request(row, arm), "repetition": repeat, "order_within_pair": arms.index(arm)}
                    records.append(current[arm])
                if mode == "paired":
                    first, second = current["generator"], current["reader"]
                    pairs.append({"id": row["id"], "repetition": repeat, "generator_seconds": first["seconds"],
                                  "reader_seconds": second["seconds"], "same_complete_input": first["context_sha256"] == second["context_sha256"] and first["question_sha256"] == second["question_sha256"],
                                  "generator_over_reader_speed_ratio": first["seconds"] / second["seconds"]})
            print(json.dumps({"stage": "benchmark_" + mode, "repetition": repeat + 1}), flush=True)
    result = {"format": "cpu-reader-benchmark-v1", "mode": mode, "created_utc": now(),
              "source_report_sha256": digest(directory / "report.json"), "policy_sha256": digest(directory / "policy.json"),
              "candidate_checkpoint": policy["diagnostic_checkpoint"], "candidate_threshold": policy["diagnostic_threshold"],
              "selected_deployment_kind": policy["kind"], "load_seconds": loading,
              "peak_process_rss_bytes": memory.peak_bytes, "peak_process_rss_gib": memory.peak_bytes / 1024**3,
              "memory_scope": "Full sampled process RSS. Paired mode loads both models; single-arm modes use separate fresh processes.",
              "request_scope": "Loaded-model tokenization, forward/generation, decoding and threshold; warmups/loading excluded. Same complete inputs, different architectures and outputs; not matched token work.",
              "selection": "First six test questions per answerability class in immutable data order; three repeats, alternating arm order, no outcome/timing selection.",
              "warmups": warmups, "records": records, "pairs": pairs,
              "median_paired_speed_ratio": statistics.median(row["generator_over_reader_speed_ratio"] for row in pairs) if pairs else None,
              "all_paired_inputs_identical": all(row["same_complete_input"] for row in pairs),
              "replay_matches": {arm: sum(row["matches_saved_batched_output"] for row in records if row["arm"] == arm) for arm in selected_arms},
              "by_arm": {arm: {"requests": len([row for row in records if row["arm"] == arm]),
                               "median_seconds": statistics.median(row["seconds"] for row in records if row["arm"] == arm),
                               "total_seconds": sum(row["seconds"] for row in records if row["arm"] == arm)} for arm in selected_arms}}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return {key: result[key] for key in ("mode", "candidate_checkpoint", "selected_deployment_kind", "by_arm", "peak_process_rss_gib", "median_paired_speed_ratio", "replay_matches")}


def answer_saved(directory, source, context, question):
    policy, _ = load_policy(directory)
    config = policy["generation_config"]
    setup(config)
    if policy["kind"] == "generator":
        model, tokenizer, _, evidence = load_generator(config, source)
        if evidence != policy["generator_source"]:
            raise ValueError("Generator source differs from saved policy.")
        answer = generator.answer(model, tokenizer, config, context, question)["answer"]
        score = None
    else:
        model, tokenizer = load_checkpoint(directory, policy, policy["checkpoint"])
        value = reader.predict_batch(model, tokenizer, [{"context": context, "question": question}],
                                     config["reader_max_length"], config["reader_max_answer_tokens"])[0]
        score = value["null_minus_span_score"]
        answer = metrics.apply_threshold(value["span_prediction"], score, policy["threshold"])
    return {"answer": answer, "device": "cpu", "selected_model": policy["kind"], "reader_checkpoint": policy["checkpoint"],
            "abstained": answer.strip() == scoring.NOT_FOUND, "null_minus_span_score": score}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/reader.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepared-data", type=Path, default=Path("artifacts/reader-data"))
    parser.add_argument("--source-run", type=Path, default=Path("artifacts/document-task-v1"))
    parser.add_argument("--prepare-model", action="store_true")
    parser.add_argument("--benchmark-run", type=Path)
    parser.add_argument("--benchmark-mode", choices=("paired", "generator", "reader"))
    parser.add_argument("--load-run", type=Path)
    parser.add_argument("--context")
    parser.add_argument("--question")
    args = parser.parse_args()
    if args.prepare_model:
        config = read(args.config)
        validate_config(config)
        path = initializer_path(config, download=True)
        result = {"reader_model_id": config["reader_model_id"], "revision": config["reader_revision"], "weights_sha256": digest(path / "model.safetensors")}
    elif args.load_run:
        if not args.context or not args.question:
            parser.error("--load-run requires --context and --question")
        result = answer_saved(args.load_run, args.source_run, args.context, args.question)
    elif args.benchmark_run:
        if not args.benchmark_mode or not args.output:
            parser.error("Benchmark requires --benchmark-mode and --output")
        result = benchmark(args.benchmark_run, args.prepared_data, args.source_run, args.benchmark_mode, args.output)
    else:
        if not args.output:
            parser.error("Experiment requires --output")
        result = run(read(args.config), args.output, args.prepared_data, args.source_run)
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
