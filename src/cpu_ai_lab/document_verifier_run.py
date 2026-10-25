"""Frozen CPU generator, learned correctness verifier, and honest cost accounting.

Generation confidence uses answer tokens excluding EOS; an EOS-only output uses
its one EOS distribution. Entropy is in nats and margin is top-1 minus top-2
probability. Features never receive reference answers or answerability labels.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import time

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import torch

from . import document_v2 as generator
from . import document_v2_metrics as scoring
from .document_task import left_pad, trim_generated
from .training import PeakRSS, adapter_snapshot, assert_cpu_model, render_prompt, write_json


def read(path):
    return json.loads(Path(path).read_bytes())


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_hash(path):
    """Canonical LF source hash survives Git's platform newline conversion."""
    return hashlib.sha256(Path(path).read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def timestamp():
    return datetime.now(timezone.utc).isoformat()


def require_new(path):
    if path.exists():
        raise FileExistsError(f"Preserving existing artifact: {path.name}")


def setup_cpu(config):
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("This experiment requires the CPU-only PyTorch build.")
    torch.set_num_threads(config["evaluation_threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(config["seed"])


def validate_config(config):
    generator.data.validate_config(config)
    for key in ("generation_batch_size", "generation_tokens", "evaluation_threads", "verifier_max_iterations",
                "benchmark_per_stratum", "benchmark_repetitions"):
        if type(config.get(key)) is not int or config[key] < 1:
            raise ValueError(f"Invalid positive integer: {key}")
    fixed = {"verifier_l2": 0.01, "minimum_correct_answer_retention": 0.95,
             "minimum_missing_answer_recall": 0.4, "maximum_answerable_f1_loss": 0.02, "threshold_grid_step": 0.01}
    if any(type(config.get(key)) not in (int, float) or config[key] != expected for key, expected in fixed.items()):
        raise ValueError("This version requires the predeclared verifier settings/gates.")


def frozen_generator(config, source):
    model, tokenizer, parameters, evidence = generator.load_source(config, source)
    model.requires_grad_(False)
    model.eval()
    assert_cpu_model(model)
    if any(p.requires_grad for p in model.parameters()):
        raise RuntimeError("The generator must remain frozen.")
    return model, tokenizer, parameters, evidence


def confidence_summary(scores, generated_ids, eos_id):
    """Summarize only distributions for this row's real answer positions."""
    ids = trim_generated(generated_ids, eos_id)
    count = len(ids) - int(bool(ids and ids[-1] == eos_id))
    used = max(1, count)
    if not ids or len(scores) < used:
        raise ValueError("Missing generated token distributions.")
    logits = torch.stack(scores[:used]).float()
    log_probs = logits.log_softmax(dim=-1)
    probs = log_probs.exp()
    selected = log_probs.gather(1, torch.tensor(ids[:used], dtype=torch.long).unsqueeze(1)).squeeze(1)
    top = probs.topk(2, dim=-1).values
    # torch.special.xlogy safely handles zero probability, unlike 0 * -inf.
    entropy = -torch.special.xlogy(probs, probs).sum(dim=-1)
    values = {"mean_logprob": float(selected.mean()), "min_logprob": float(selected.min()),
              "mean_entropy": float(entropy.mean()), "mean_margin": float((top[:, 0] - top[:, 1]).mean()),
              "generated_tokens": len(ids), "ended_with_eos": bool(ids[-1] == eos_id)}
    if not all(math.isfinite(float(value)) for value in values.values()):
        raise ValueError("Nonfinite confidence feature.")
    return values


@torch.inference_mode()
def generate_batch(model, tokenizer, rows, token_limit, with_confidence=True):
    """Tokenization through decoded answer/features; no evaluation labels read."""
    from .document_verifier import features
    prompts = [tokenizer.encode(render_prompt(tokenizer, {"prompt": row["prompt"]}), add_special_tokens=False) for row in rows]
    batch = left_pad(prompts, tokenizer.pad_token_id)
    options = dict(do_sample=False, use_cache=True, max_new_tokens=token_limit,
                   eos_token_id=tokenizer.eos_token_id, pad_token_id=tokenizer.pad_token_id)
    if with_confidence:
        options.update(return_dict_in_generate=True, output_scores=True)
    output = model.generate(**batch, **options)
    sequences = output.sequences if with_confidence else output
    width = batch["input_ids"].shape[1]
    result = []
    for index, row in enumerate(rows):
        ids = trim_generated(sequences[index, width:].tolist(), tokenizer.eos_token_id)
        prediction = tokenizer.decode(ids, skip_special_tokens=True)
        item = {"id": row.get("id"), "prediction": prediction,
                "prediction_sha256": scoring.sha256(prediction),
                "generated_tokens_including_eos": len(ids), "ended_with_eos": bool(ids and ids[-1] == tokenizer.eos_token_id)}
        if with_confidence:
            confidence = confidence_summary([step[index] for step in output.scores], ids, tokenizer.eos_token_id)
            item.update(confidence=confidence, features=features(row["context"], row["question"], prediction, confidence))
        result.append(item)
    return result


def collect(model, tokenizer, rows, config, split):
    # Group by known source prompt lengths; restore immutable data order.
    order = sorted(range(len(rows)), key=lambda i: (rows[i]["prompt_tokens"], i))
    local = [None] * len(rows)
    batch_size = config["generation_batch_size"]
    started = time.perf_counter()
    for start in range(0, len(order), batch_size):
        indices = order[start:start + batch_size]
        values = generate_batch(model, tokenizer, [rows[i] for i in indices], config["generation_tokens"])
        for index, value in zip(indices, values):
            local[index] = value
        if start % (batch_size * 8) == 0 or start + batch_size >= len(rows):
            print(json.dumps({"stage": "generate_" + split, "processed": min(start + batch_size, len(rows)), "total": len(rows)}), flush=True)
    return local, time.perf_counter() - started


def metric(rows, predictions):
    records = [scoring.score_prediction(row, prediction) for row, prediction in zip(rows, predictions)]
    if len(rows) != len(predictions):
        raise ValueError("Prediction count differs.")
    return {**scoring.summarize(records), "records": records}


def public_features(rows, local, probabilities):
    return [{"id": row["id"], "prediction_sha256": value["prediction_sha256"],
             "confidence": value["confidence"], "features": value["features"], "verifier_probability": float(probability)}
            for row, value, probability in zip(rows, local, probabilities)]


def fixed_benchmark_rows(rows, per_stratum=3):
    groups = {name: 0 for name in generator.data.STRATA}
    chosen = []
    for row in rows:
        group = generator.data.stratum(row)
        if groups[group] < per_stratum:
            chosen.append(row)
            groups[group] += 1
    if any(count != per_stratum for count in groups.values()):
        raise ValueError("Insufficient examples for fixed benchmark selection.")
    return chosen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/document-verifier.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--prepared-data", type=Path, default=Path("artifacts/document-verifier-data"))
    parser.add_argument("--source-run", type=Path, default=Path("artifacts/document-task-v1"))
    parser.add_argument("--benchmark-run", type=Path)
    parser.add_argument("--benchmark-mode", choices=("paired", "baseline", "verifier"))
    parser.add_argument("--load-policy", type=Path)
    parser.add_argument("--context")
    parser.add_argument("--question")
    args = parser.parse_args()
    if args.load_policy:
        if not args.context or not args.question:
            parser.error("--load-policy requires --context and --question")
        result = answer_saved(args.load_policy, args.source_run, args.context, args.question)
    elif args.benchmark_run:
        if not args.benchmark_mode or not args.output:
            parser.error("Benchmark requires --benchmark-mode and --output")
        result = benchmark(args.benchmark_run, args.prepared_data, args.source_run, args.benchmark_mode, args.output)
    else:
        if not args.output:
            parser.error("Experiment requires --output")
        result = run(read(args.config), args.output, args.prepared_data, args.source_run)
    print(json.dumps(result, indent=2, allow_nan=False))


# Experiment and deployment functions below keep all test inference after the
# validation policy has been serialized. This ordering is part of the contract.


def candidate_threshold(selection):
    if selection["accepted"]:
        return selection["selected_threshold"]
    trial = selection["best_constrained_candidate"]
    return trial["threshold"] if trial else 0.0


def policy_metrics(rows, local, probabilities, selection):
    from . import document_verifier as verifier
    from .document_guard import guard_output
    predictions = [row["prediction"] for row in local]
    confidence = [math.exp(row["confidence"]["mean_logprob"]) for row in local]
    return {
        "baseline": metric(rows, predictions),
        "verifier_candidate": verifier.evaluate_policy(rows, predictions, probabilities, candidate_threshold(selection["verifier"])),
        "selected": verifier.evaluate_policy(rows, predictions, probabilities, selection["verifier"]["selected_threshold"], enabled=selection["verifier"]["selected_enabled"]),
        "confidence_candidate": verifier.evaluate_policy(rows, predictions, confidence, candidate_threshold(selection["confidence"])),
        "confidence_selected": verifier.evaluate_policy(rows, predictions, confidence, selection["confidence"]["selected_threshold"], enabled=selection["confidence"]["selected_enabled"]),
        "span_guard": metric(rows, [guard_output(row["context"], prediction)["answer"] for row, prediction in zip(rows, predictions)]),
    }


def run(config, output, prepared, source):
    from . import document_verifier as verifier
    from . import document_verifier_data as data
    validate_config(config)
    require_new(output)
    output.mkdir(parents=True)
    write_json(output / "resolved_config.json", config)
    write_json(output / "run_status.json", {"status": "running", "started_utc": timestamp()})
    setup_cpu(config)
    started = time.perf_counter()
    try:
        with PeakRSS() as memory:
            splits, provenance = data.load_prepared(prepared, config)
            write_json(output / "data_provenance.json", provenance)
            (output / "split_manifest.json").write_bytes((prepared / "split_manifest.json").read_bytes())
            model, tokenizer, parameters, source_evidence = frozen_generator(config, source)
            if provenance["prior_manifest_sha256"][0] != source_evidence["split_manifest_sha256"]:
                raise ValueError("Prepared ancestor exclusion manifest differs from the frozen generator's source.")
            initial_state = generator.state_hash(adapter_snapshot(model))
            # Persist recipe and implementation fingerprints before any generations.
            protocol = {"format": "cpu-document-verifier-protocol-v1", "created_utc": timestamp(),
                        "config": config, "source": source_evidence, "feature_names": list(verifier.FEATURE_NAMES),
                        "feature_definition": "Fixed lexical features plus processed greedy-generation confidence; answer positions exclude EOS except EOS-only outputs. Entropy in nats, margin top1-top2 probability.",
                        "prompt_instruction": generator.data.INSTRUCTION,
                        "label": "1 only for exact-correct answerable raw model outputs; all missing-answer and incorrect answerable outputs are 0",
                        "selection": "One L2 logistic fit on train only. Threshold grid and gates use validation only; test generation occurs after policy serialization.",
                        "software_sha256": {name: code_hash(Path(__file__).with_name(name)) for name in ("document_verifier_run.py", "document_verifier.py", "document_verifier_data.py", "document_v2_data.py", "document_v2_metrics.py")},
                        "software_hash_encoding": "UTF-8 with universal newlines normalized to LF",
                        "generator_trainable_parameters": 0,
                        "timing": "Pipeline includes model load, feature-generation, classifier fit, selection and test; excludes interpreter startup, prepared dataset creation and final report serialization."}
            write_json(output / "protocol.json", protocol)
            stage_seconds, local = {}, {}
            for name in ("train", "validation"):
                local[name], stage_seconds[name + "_generation_and_features"] = collect(model, tokenizer, splits[name], config, name)
                write_json(output / (name + "-local.json"), local[name])
            training_scores = metric(splits["train"], [row["prediction"] for row in local["train"]])
            labels = [int(not row["is_impossible"] and row["exact_match"] == 1) for row in training_scores["records"]]
            fit_started = time.perf_counter()
            classifier = verifier.fit([row["features"] for row in local["train"]], labels,
                                      l2=config["verifier_l2"], max_iter=config["verifier_max_iterations"])
            stage_seconds["classifier_fit"] = time.perf_counter() - fit_started
            verifier.save_model(output / "verifier.json", classifier)
            loaded_classifier = verifier.load_model(output / "verifier.json")
            probabilities = {name: verifier.predict(classifier, [row["features"] for row in local[name]]) for name in ("train", "validation")}
            reloaded_probs = verifier.predict(loaded_classifier, [row["features"] for row in local["validation"]])
            reload_delta = max(abs(float(a) - float(b)) for a, b in zip(probabilities["validation"], reloaded_probs))
            if reload_delta != 0:
                raise RuntimeError("Saved classifier did not reproduce validation scores exactly.")
            tick = time.perf_counter()
            predictions = [row["prediction"] for row in local["validation"]]
            selection = {
                "format": "cpu-document-verifier-selection-v1", "completed_utc": timestamp(), "uses_test": False,
                "verifier": verifier.select_threshold(splits["validation"], predictions, probabilities["validation"]),
                "confidence": verifier.select_threshold(splits["validation"], predictions, [math.exp(row["confidence"]["mean_logprob"]) for row in local["validation"]]),
            }
            stage_seconds["validation_selection"] = time.perf_counter() - tick
            write_json(output / "selection.json", selection)
            policy = {"format": "cpu-document-verifier-policy-v1", "generation_config": config,
                      "source": source_evidence, "verifier_sha256": file_hash(output / "verifier.json"),
                      "selection_sha256": file_hash(output / "selection.json"),
                      "protocol_sha256": file_hash(output / "protocol.json"),
                      "enabled": selection["verifier"]["selected_enabled"],
                      "threshold": selection["verifier"]["selected_threshold"],
                      "diagnostic_candidate_threshold": candidate_threshold(selection["verifier"]),
                      "selection_uses_test": False, "created_utc": timestamp()}
            write_json(output / "policy.json", policy)
            frozen_policy_hash = file_hash(output / "policy.json")
            print(json.dumps({"stage": "selection_frozen_before_test", "accepted": selection["verifier"]["accepted"],
                              "threshold": policy["threshold"], "diagnostic_threshold": policy["diagnostic_candidate_threshold"]}), flush=True)
            test_started_utc = timestamp()
            local["test"], stage_seconds["test_generation_and_features"] = collect(model, tokenizer, splits["test"], config, "test")
            write_json(output / "test-local.json", local["test"])
            probabilities["test"] = verifier.predict(classifier, [row["features"] for row in local["test"]])
            evaluations = {name: policy_metrics(splits[name], local[name], probabilities[name], selection) for name in ("validation", "test")}
            for name in ("train", "validation", "test"):
                write_json(output / (name + "-features.json"), public_features(splits[name], local[name], probabilities[name]))
            final_state = generator.state_hash(adapter_snapshot(model))
            if initial_state != final_state or file_hash(output / "policy.json") != frozen_policy_hash:
                raise RuntimeError("Frozen generator or selected policy changed during evaluation.")
            elapsed = time.perf_counter() - started
        report = {"format": "cpu-document-verifier-report-v1", "completed_utc": timestamp(),
                  "source": source_evidence, "dataset": provenance, "training": {
                      "examples": len(labels), "positive_correct_answers": sum(labels), "negative_answers": len(labels) - sum(labels),
                      "train_baseline": {k: v for k, v in training_scores.items() if k != "records"},
                      "classifier_parameters": len(verifier.FEATURE_NAMES) + 1, "generator_parameters": parameters,
                      "generator_trainable_parameters": 0, "classifier_reload_max_probability_difference": reload_delta},
                  "cpu": {"torch_version": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(),
                          "parameter_devices": sorted({str(p.device) for p in model.parameters()}),
                          "initial_adapter_sha256": initial_state, "final_adapter_sha256": final_state,
                          "numpy_classifier_runs_on": "CPU"},
                  "timing": {"seconds": elapsed, "stages_seconds": stage_seconds,
                             "peak_process_rss_bytes": memory.peak_bytes, "peak_process_rss_gib": memory.peak_bytes / 1024**3},
                  "selection": {"accepted": selection["verifier"]["accepted"], "enabled": policy["enabled"],
                                "threshold": policy["threshold"], "diagnostic_candidate_threshold": policy["diagnostic_candidate_threshold"],
                                "policy_saved_utc": policy["created_utc"], "test_generation_started_utc": test_started_utc,
                                "test_used_for_selection": False},
                  **evaluations,
                  "limitations": ["Filtered balanced historical SQuAD2 paragraphs, not production prevalence or unseen topics.",
                                  "The classifier score is uncalibrated; 50/50 answerability sampling differs from user queries.",
                                  "Only the small correctness classifier is trained; the pretrained language model and its document adapter remain frozen.",
                                  "A verifier abstention is an estimated error, not proof that the passage has no answer."]}
        public_files = ("resolved_config.json", "protocol.json", "selection.json", "policy.json", "verifier.json", "data_provenance.json",
                        "split_manifest.json", "train-features.json", "validation-features.json", "test-features.json")
        report["artifact_sha256"] = {name: file_hash(output / name) for name in public_files}
        report["local_prediction_sha256"] = {name + "-local.json": file_hash(output / (name + "-local.json")) for name in local}
        write_json(output / "report.json", report)
        write_json(output / "run_status.json", {"status": "complete", "completed_utc": timestamp()})
        return {"selection": report["selection"], "timing": report["timing"],
                "test": {name: {k: v for k, v in value.items() if k not in ("records", "by_stratum", "by_length")} for name, value in report["test"].items()}}
    except BaseException as error:
        write_json(output / "run_status.json", {"status": "failed", "error_type": type(error).__name__})
        raise


def load_policy(path):
    from . import document_verifier as verifier
    policy = read(path)
    if policy.get("format") != "cpu-document-verifier-policy-v1" or policy.get("selection_uses_test") is not False:
        raise ValueError("Unsupported policy contract.")
    directory = path.parent
    for name in ("verifier", "selection", "protocol"):
        if file_hash(directory / (name + ".json")) != policy[name + "_sha256"]:
            raise ValueError(f"Policy {name} checksum mismatch.")
    selection, protocol = read(directory / "selection.json"), read(directory / "protocol.json")
    if policy["generation_config"] != protocol["config"] or policy["source"] != protocol["source"]:
        raise ValueError("Policy configuration/source differs from its frozen protocol.")
    if protocol["prompt_instruction"] != generator.data.INSTRUCTION:
        raise ValueError("Current prompt differs from the frozen protocol.")
    for name in ("document_verifier_run.py", "document_verifier.py", "document_verifier_data.py", "document_v2_data.py", "document_v2_metrics.py"):
        if protocol["software_sha256"].get(name) != code_hash(Path(__file__).with_name(name)):
            raise ValueError("Current implementation differs from the frozen protocol.")
    validate_config(policy["generation_config"])
    if type(policy["enabled"]) is not bool or selection.get("uses_test") is not False:
        raise ValueError("Invalid validation-only selection contract.")
    if (policy["enabled"] != selection["verifier"]["selected_enabled"] or
        policy["threshold"] != selection["verifier"]["selected_threshold"] or
        policy["diagnostic_candidate_threshold"] != candidate_threshold(selection["verifier"])):
        raise ValueError("Policy differs from validation selection.")
    return policy, verifier.load_model(directory / "verifier.json")


def answer_saved(path, source, context, question):
    from . import document_verifier as verifier
    policy, classifier = load_policy(path)
    config = policy["generation_config"]
    setup_cpu(config)
    model, tokenizer, _, evidence = frozen_generator(config, source)
    if evidence != policy["source"]:
        raise ValueError("Model source differs from the frozen policy.")
    row = {"prompt": generator.data.prompt_text(context, question), "context": context, "question": question}
    if len(tokenizer.encode(context, add_special_tokens=False)) > config["max_context_tokens"] or len(tokenizer.encode(render_prompt(tokenizer, row), add_special_tokens=False)) > config["max_prompt_tokens"]:
        raise ValueError("Passage/prompt exceeds this experiment's complete-context budget.")
    value = generate_batch(model, tokenizer, [row], config["generation_tokens"], with_confidence=policy["enabled"])[0]
    probability = verifier.predict(classifier, [value["features"]])[0] if policy["enabled"] else None
    answer = verifier.apply_policy(value["prediction"], probability if probability is not None else 1.0, policy["threshold"], enabled=policy["enabled"])
    return {"answer": answer, "device": "cpu", "verifier_enabled": policy["enabled"], "verifier_score": probability,
            "abstention_origin": ("model" if value["prediction"].strip() == scoring.NOT_FOUND else "verifier") if answer.strip() == scoring.NOT_FOUND else None}


def benchmark(directory, prepared, source, mode, output):
    """Paired warm request latency; independent single-arm processes for RSS."""
    from . import document_verifier as verifier
    from . import document_verifier_data as data
    require_new(output)
    if read(directory / "run_status.json").get("status") != "complete":
        raise ValueError("Benchmark requires a completed experiment.")
    report = read(directory / "report.json")
    for name, expected in report["artifact_sha256"].items():
        if file_hash(directory / name) != expected:
            raise ValueError("Completed report artifact checksum mismatch.")
    policy, classifier = load_policy(directory / "policy.json")
    config = policy["generation_config"]
    setup_cpu(config)
    splits, provenance = data.load_prepared(prepared, config)
    if provenance != report["dataset"]:
        raise ValueError("Benchmark data differs from experiment data.")
    rows = fixed_benchmark_rows(splits["test"], config["benchmark_per_stratum"])
    expected_raw = {r["id"]: r for r in report["test"]["baseline"]["records"]}
    expected_candidate = {r["id"]: r for r in report["test"]["verifier_candidate"]["records"]}
    expected_features = {r["id"]: r for r in read(directory / "test-features.json")}
    started = time.perf_counter()
    with PeakRSS() as memory:
        model, tokenizer, _, evidence = frozen_generator(config, source)
        if evidence != policy["source"]:
            raise ValueError("Benchmark generator source differs.")
        load_seconds = time.perf_counter() - started

        def request(row, arm):
            tick = time.perf_counter()
            value = generate_batch(model, tokenizer, [row], config["generation_tokens"], with_confidence=arm == "verifier")[0]
            probability = None
            answer = value["prediction"]
            if arm == "verifier":
                probability = float(verifier.predict(classifier, [value["features"]])[0])
                answer = verifier.apply_policy(answer, probability, policy["diagnostic_candidate_threshold"])
            elapsed = time.perf_counter() - tick
            return {"id": row["id"], "stratum": generator.data.stratum(row), "arm": arm, "seconds": elapsed,
                    "raw_prediction_sha256": value["prediction_sha256"], "prediction_sha256": scoring.sha256(answer),
                    "generated_tokens_including_eos": value["generated_tokens_including_eos"],
                    "verifier_probability": probability,
                    "matches_saved_batched_generator": value["prediction_sha256"] == expected_raw[row["id"]]["prediction_sha256"],
                    "matches_saved_batched_candidate": scoring.sha256(answer) == expected_candidate[row["id"]]["prediction_sha256"] if arm == "verifier" else None,
                    "probability_delta_from_saved_batch": abs(probability - expected_features[row["id"]]["verifier_probability"]) if probability is not None else None}

        warmups = [request(rows[0], arm) for arm in (("baseline", "verifier") if mode == "paired" else (mode,))]
        records, pairs = [], []
        repetitions = config["benchmark_repetitions"] if mode == "paired" else 1
        for repeat in range(repetitions):
            for index, row in enumerate(rows):
                arms = ("baseline", "verifier") if (repeat + index) % 2 == 0 else ("verifier", "baseline")
                if mode != "paired":
                    arms = (mode,)
                current = {}
                for arm in arms:
                    result = {**request(row, arm), "repetition": repeat, "order_within_pair": arms.index(arm)}
                    records.append(result)
                    current[arm] = result
                if mode == "paired":
                    first, second = current["baseline"], current["verifier"]
                    matched = (first["raw_prediction_sha256"] == second["raw_prediction_sha256"] and
                               first["generated_tokens_including_eos"] == second["generated_tokens_including_eos"])
                    pairs.append({"id": row["id"], "repetition": repeat, "matched_work": matched,
                                  "baseline_seconds": first["seconds"], "verifier_seconds": second["seconds"],
                                  "verifier_over_baseline_latency_ratio": second["seconds"] / first["seconds"] if matched else None})
            print(json.dumps({"stage": "benchmark_" + mode, "completed_repetitions": repeat + 1}), flush=True)
    matched = [row for row in pairs if row["matched_work"]]
    result = {"format": "cpu-document-verifier-benchmark-v1", "mode": mode, "created_utc": timestamp(),
              "source_report_sha256": file_hash(directory / "report.json"), "policy_sha256": file_hash(directory / "policy.json"),
              "selection": "First three test records per stratum in immutable source order; not selected by correctness or timing.",
              "timing_scope": "Warm loaded-model request: tokenize, generate, decode; verifier additionally collects distributions, extracts features, scores and applies fixed diagnostic threshold. Model loading and warmups reported separately.",
              "candidate_enabled_for_cost_measurement": True, "selected_deployment_verifier_enabled": policy["enabled"],
              "evaluation_threads": config["evaluation_threads"], "load_seconds": load_seconds,
              "peak_process_rss_bytes": memory.peak_bytes, "peak_process_rss_gib": memory.peak_bytes / 1024**3,
              "memory_scope": "Sampled full process RSS, including model and runtime; paired mode shares allocator across arms, single-arm modes run in fresh processes.",
              "warmups": warmups, "records": records, "pairs": pairs,
              "raw_generator_reload_matches": sum(row["matches_saved_batched_generator"] for row in records),
              "candidate_reload_matches": sum(row["matches_saved_batched_candidate"] is True for row in records),
              "matched_pairs": len(matched), "total_pairs": len(pairs),
              "median_paired_latency_ratio": statistics.median(row["verifier_over_baseline_latency_ratio"] for row in matched) if matched else None,
              "by_arm": {arm: {"requests": len([row for row in records if row["arm"] == arm]),
                               "median_seconds": statistics.median(row["seconds"] for row in records if row["arm"] == arm),
                               "total_seconds": sum(row["seconds"] for row in records if row["arm"] == arm)}
                         for arm in ("baseline", "verifier") if any(row["arm"] == arm for row in records)}}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return {key: result[key] for key in ("mode", "by_arm", "peak_process_rss_gib", "matched_pairs", "total_pairs", "median_paired_latency_ratio", "raw_generator_reload_matches", "candidate_reload_matches")}


if __name__ == "__main__":
    main()
