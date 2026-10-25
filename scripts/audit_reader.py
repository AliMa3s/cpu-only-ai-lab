"""Audit completed CPU reader evidence without importing Torch or running a model.

Run after all timed experiments and benchmarks finish. The audit independently
replays thresholds, paired retention and checkpoint choice; verifies native
source data, schedules, checkpoint tensor bytes and saved output scores; and
checks public publication hashes. Tokenizer-only alignment is allowed here.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
import mmap
from pathlib import Path
import random
import re
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from cpu_ai_lab import reader_data as data
from cpu_ai_lab import document_v2_metrics as scoring

GRID = ["always_answer"] + [index / 2 for index in range(-40, 41)] + ["always_abstain"]
SOFTWARE = {"reader_run.py", "reader_data.py", "reader_model.py", "reader_metrics.py",
            "document_v2.py", "document_v2_data.py", "document_v2_metrics.py"}
PUBLIC = {"resolved_config.json", "protocol.json", "selection.json", "policy.json", "data_provenance.json",
          "split_manifest.json", "sample_schedule.json", "training_curve.json", "generator-validation.json",
          "generator-test.json", "initial-validation-decoded.json", "final-validation-decoded.json",
          "initial-test-decoded.json", "final-test-decoded.json"}
BENCHMARKS = {"benchmark-paired.json", "benchmark-generator.json", "benchmark-reader.json"}
RETENTION_DEFINITION = "Among answerable questions exactly correct for the generator, the fraction also exactly correct for this candidate; newly correct questions never offset lost correct questions."


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(value):
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def file_hash(path):
    checksum = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


class Audit:
    def __init__(self):
        self.checks = 0

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            raise ValueError("Audit failed: " + label)

    def equal(self, actual, expected, label, tolerance=1e-10):
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict) and set(actual) == set(expected), label + " keys")
            for key in expected:
                self.equal(actual[key], expected[key], label + "/" + key, tolerance)
        elif isinstance(expected, list):
            self.check(isinstance(actual, list) and len(actual) == len(expected), label + " length")
            for first, second in zip(actual, expected, strict=True):
                self.equal(first, second, label, tolerance)
        elif type(expected) in (int, float) and type(actual) in (int, float):
            self.check(math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), label)
        else:
            self.check(type(actual) is type(expected) and actual == expected, label)

    def checked_read(self, path, expected):
        self.equal(file_hash(path), expected, Path(path).name + " checksum")
        return read(path)


def paired_retention(baseline, candidate):
    original = {row["id"] for row in baseline if not row["is_impossible"] and row["exact_match"] == 1}
    current = {row["id"] for row in candidate if not row["is_impossible"] and row["exact_match"] == 1}
    retained = original & current
    return {"baseline_exact_correct": len(original), "retained_exact_correct": len(retained),
            "lost_exact_correct": len(original - current), "new_exact_correct": len(current - original),
            "candidate_exact_correct": len(current), "exact_correct_retention": len(retained) / len(original) if original else None,
            "definition": RETENTION_DEFINITION}


def baseline_metric(rows, predictions):
    records = []
    for row, prediction in zip(rows, predictions, strict=True):
        scored = scoring.score_prediction(row, prediction)
        correct = not row["is_impossible"] and scored["exact_match"] == 1
        records.append({**scored, "generator_prediction_sha256": scored["prediction_sha256"],
                        "generator_exact_correct_answerable": correct, "retained_generator_exact_correct": correct,
                        "newly_exact_correct_answerable": False})
    return {**scoring.summarize(records), "paired_retention": paired_retention(records, records), "records": records}


def reader_metric(rows, decoded, threshold, baseline):
    records = []
    for row, output, previous in zip(rows, decoded, baseline["records"], strict=True):
        withheld = threshold == "always_abstain" or (threshold != "always_answer" and output["null_minus_span_score"] > threshold)
        prediction = "NOT_FOUND" if withheld else output["span_prediction"]
        scored = scoring.score_prediction(row, prediction)
        old_correct = not row["is_impossible"] and previous["exact_match"] == 1
        current_correct = not row["is_impossible"] and scored["exact_match"] == 1
        records.append({**scored, "reader_span_sha256": digest(output["span_prediction"]),
                        "null_minus_span_score": float(output["null_minus_span_score"]), "threshold_withheld": withheld,
                        "generator_prediction_sha256": previous["prediction_sha256"],
                        "generator_exact_correct_answerable": old_correct,
                        "retained_generator_exact_correct": old_correct and current_correct,
                        "newly_exact_correct_answerable": not old_correct and current_correct})
    return {**scoring.summarize(records), "paired_retention": paired_retention(baseline["records"], records),
            "records": records, "threshold": threshold}


def gate(baseline, candidate):
    paired = paired_retention(baseline["records"], candidate["records"])
    old, retained = paired["baseline_exact_correct"], paired["retained_exact_correct"]
    checks = {"positive_baseline_exact_correct_count": old > 0,
              "paired_exact_correct_retention_at_least_0_95": old > 0 and retained >= 0.95 * old,
              "answerable_f1_drop_at_most_0_02": candidate["answerable_token_f1"] >= baseline["answerable_token_f1"] - 0.02,
              "missing_recall_at_least_0_40": candidate["correct_abstention_recall"] >= 0.4}
    return {"accepted": all(checks.values()), "checks": checks, **paired,
            "answerable_f1_change": candidate["answerable_token_f1"] - baseline["answerable_token_f1"],
            "retention_and_f1_pass": all(value for key, value in checks.items() if key != "missing_recall_at_least_0_40")}


def audit_thresholds(audit, saved, rows, decoded, baseline):
    audit.equal(saved["baseline"], baseline, "validation generator comparator")
    audit.equal(saved["criteria"]["threshold_grid"], GRID, "fixed threshold grid")
    audit.check(len(saved["trials"]) == len(GRID), "exact threshold trial count")
    passing, constrained, all_trials = [], [], []
    for index, (threshold, stored) in enumerate(zip(GRID, saved["trials"], strict=True)):
        metric = reader_metric(rows, decoded, threshold, baseline)
        quality = gate(baseline, metric)
        trial = {"threshold": threshold, "metrics": {key: value for key, value in metric.items() if key != "records"}, "gate": quality}
        audit.equal(stored, trial, "independent validation trial and paired gate")
        key = (metric["correct_abstention_recall"], metric["answerable_token_f1"], metric["answerable_exact_match"], metric["coverage"], -index)
        if quality["accepted"]:
            passing.append((key, trial, metric))
        if quality["retention_and_f1_pass"]:
            constrained.append((key, trial))
        all_trials.append(trial)
    selected = max(passing, key=lambda item: item[0]) if passing else None
    best = max(constrained, key=lambda item: item[0])[1] if constrained else None
    diagnostic = best if best is not None else all_trials[0]
    audit.equal(saved["accepted"], bool(selected), "threshold acceptance")
    audit.equal(saved["selected_enabled"], bool(selected), "threshold enabled")
    audit.equal(saved["selected_kind"], "reader" if selected else "generator", "threshold fallback kind")
    audit.equal(saved["selected_threshold"], selected[1]["threshold"] if selected else None, "selected threshold")
    audit.equal(saved["selected"], selected[2] if selected else baseline, "selected threshold metrics")
    audit.equal(saved["selected_gate"], selected[1]["gate"] if selected else None, "selected gate")
    audit.equal(saved["best_constrained_candidate"], best, "constrained candidate")
    audit.equal(saved["diagnostic_candidate"], diagnostic, "diagnostic candidate")
    audit.equal(saved["diagnostic_threshold"], diagnostic["threshold"], "diagnostic threshold")
    audit.equal(saved["diagnostic_satisfies_quality_constraints"], best is not None, "diagnostic gate status")
    audit.equal(saved["validation_identity_sha256"], baseline["evaluation_identity_sha256"], "validation identity")
    audit.check(saved["test_used_for_selection"] is False, "no test threshold selection")


def choose_checkpoint(selections):
    accepted = [name for name in ("initial", "final") if selections[name]["accepted"]]
    if accepted:
        name = max(accepted, key=lambda item: (selections[item]["selected"]["answerable_token_f1"], selections[item]["selected"]["correct_abstention_recall"], item == "initial"))
        threshold = selections[name]["selected_threshold"]
        return {"kind": "reader", "checkpoint": name, "threshold": threshold, "diagnostic_checkpoint": name, "diagnostic_threshold": threshold}
    name = max(("initial", "final"), key=lambda item: (selections[item]["diagnostic_candidate"]["gate"]["retention_and_f1_pass"],
                selections[item]["diagnostic_candidate"]["gate"]["exact_correct_retention"] or 0.0,
                selections[item]["diagnostic_candidate"]["metrics"]["answerable_token_f1"],
                selections[item]["diagnostic_candidate"]["metrics"]["correct_abstention_recall"], item == "initial"))
    return {"kind": "generator", "checkpoint": None, "threshold": None, "diagnostic_checkpoint": name, "diagnostic_threshold": selections[name]["diagnostic_threshold"]}


def tensor_state(path):
    """Reproduce sorted-name/state tensor SHA from safetensors bytes, no model load."""
    checksum, parameters = hashlib.sha256(), 0
    with Path(path).open("rb") as handle, mmap.mmap(handle.fileno(), 0, access=mmap.ACCESS_READ) as payload:
        size = int.from_bytes(payload[:8], "little")
        header = json.loads(payload[8:8 + size])
        for name in sorted(key for key in header if key != "__metadata__"):
            tensor = header[name]
            if tensor["dtype"] != "F32":
                raise ValueError("Audit expects the experiment's float32 reader tensors.")
            start, end = tensor["data_offsets"]
            checksum.update(name.encode("utf-8"))
            checksum.update(payload[8 + size + start:8 + size + end])
            parameters += math.prod(tensor["shape"])
    return checksum.hexdigest(), parameters


def audit_checkpoint(audit, directory, stage, report, policy):
    folder = directory / (stage + "_reader")
    manifest = audit.checked_read(folder / "reader_manifest.json", policy["reader_manifest_sha256"][stage])
    audit.equal(manifest, report["reader_manifests"][stage], stage + " report checkpoint binding")
    content = {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    audit.equal(manifest["manifest_sha256"], digest(json.dumps(content, sort_keys=True, separators=(",", ":"), allow_nan=False)), stage + " manifest integrity")
    files = manifest["files_sha256"]
    audit.equal({path.name for path in folder.iterdir() if path.is_file() and path.name != "reader_manifest.json"}, set(files), stage + " checkpoint file allowlist")
    for name, expected in files.items():
        audit.check(Path(name).name == name and "/" not in name and "\\" not in name, "checkpoint path stays local")
        audit.equal(file_hash(folder / name), expected, stage + " checkpoint file " + name)
    state, parameters = tensor_state(folder / "model.safetensors")
    audit.equal(state, report["training"][stage + "_state_sha256"], stage + " independently reconstructed tensor-state hash")
    audit.equal(parameters, report["training"]["trainable_parameters"], stage + " actual tensor parameter count")
    return manifest


def audit_generator(audit, rows, local, stored):
    records = []
    audit.equal([row["id"] for row in local], [row["id"] for row in rows], "generator source order")
    for row, output in zip(rows, local, strict=True):
        score = scoring.score_prediction(row, output["prediction"])
        record = {**score, "generated_tokens_including_eos": output["generated_tokens_including_eos"], "ended_with_eos": output["ended_with_eos"]}
        audit.equal({key: value for key, value in output.items() if key not in ("question", "references", "prediction")}, record, "every raw generator score")
        audit.equal(output["question"], row["question"], "generator question integrity")
        audit.equal(output["references"], row["references"], "generator complete gold references")
        records.append(record)
    expected = {**scoring.summarize(records), "generated_tokens_including_eos": sum(row["generated_tokens_including_eos"] for row in records),
                "ended_with_eos": sum(row["ended_with_eos"] for row in records), "records": records}
    audit.equal({key: value for key, value in stored.items() if key != "seconds"}, expected, "generator aggregate")
    audit.check(stored["seconds"] > 0, "positive generation duration")


def audit_decoded(audit, rows, local, public, encodings, config):
    audit.equal([row["id"] for row in local], [row["id"] for row in rows], "reader source order")
    expected_public = []
    for row, output in zip(rows, local, strict=True):
        encoded = encodings[row["id"]]
        start, end = output["start_token"], output["end_token"]
        audit.check(0 <= start <= end < len(encoded.ids) and end - start + 1 <= config["reader_max_answer_tokens"], "decoded token span budget")
        audit.check(all(encoded.sequence_ids[index] == 1 for index in range(start, end + 1)), "decoded span lies wholly in context")
        audit.equal([output["start_char"], output["end_char"]], [encoded.offsets[start][0], encoded.offsets[end][1]], "decoded exact token boundaries")
        audit.equal(output["span_prediction"], row["context"][output["start_char"]:output["end_char"]], "decoded literal full-context substring")
        audit.equal(output["input_tokens"], len(encoded.ids), "decoded complete input length")
        audit.equal(output["null_minus_span_score"], output["null_score"] - output["best_span_score"], "null-minus-best-span arithmetic")
        audit.check(all(math.isfinite(output[key]) for key in ("null_score", "best_span_score", "null_minus_span_score")), "finite saved decoder scores")
        expected_public.append({**{key: value for key, value in output.items() if key != "span_prediction"}, "span_prediction_sha256": digest(output["span_prediction"])})
    audit.equal(public, expected_public, "hashed public decoded outputs")


def audit_schedule(audit, rows, schedule, curve, report, config):
    rng = random.Random(config["seed"])
    groups = [[index for index, row in enumerate(rows) if row["is_impossible"] == missing] for missing in (False, True)]
    for group in groups:
        rng.shuffle(group)
    expected, half = [], config["training_batch_size"] // 2
    for start in range(0, len(groups[0]), half):
        batch = groups[0][start:start + half] + groups[1][start:start + half]
        rng.shuffle(batch)
        expected.append([rows[index]["id"] for index in batch])
    audit.equal(schedule, expected, "independent one-pass balanced schedule")
    audit.equal(len(schedule), config["optimizer_steps"], "planned update count")
    by_id = {row["id"]: row for row in rows}
    audit.equal(Counter(identifier for batch in schedule for identifier in batch), Counter(by_id.keys()), "every training row scheduled once")
    for batch in schedule:
        audit.check(len(batch) == config["training_batch_size"] and sum(by_id[identifier]["is_impossible"] for identifier in batch) == half, "each batch has equal answerability")
    audit.equal(len(curve), report["training"]["completed_updates"], "actual recorded updates")
    audit.check(0 < len(curve) <= len(schedule), "nonzero completed training")
    for step, item in enumerate(curve, 1):
        batch = schedule[step - 1]
        rate = config["learning_rate"] * (step / config["warmup_steps"] if step <= config["warmup_steps"] else (config["optimizer_steps"] - step + 1) / (config["optimizer_steps"] - config["warmup_steps"]))
        audit.equal(item["step"], step, "monotonic updates")
        audit.equal(item["learning_rate"], rate, "fixed warmup/decay schedule")
        audit.equal(item["input_tokens"], sum(by_id[identifier]["reader_tokens"] for identifier in batch), "actual training token accounting")
        audit.equal(item["padded_tokens"], len(batch) * max(by_id[identifier]["reader_tokens"] for identifier in batch), "actual padded token accounting")
        audit.check(all(math.isfinite(item[key]) and item[key] >= 0 for key in ("loss", "gradient_norm_before_clipping", "compute_seconds")), "finite training measurements")
    seen = {identifier for batch in schedule[:len(curve)] for identifier in batch}
    for key, expected_value in (("unique_examples_seen", len(seen)), ("presentations", len(curve) * config["training_batch_size"]),
                                ("training_tokens", sum(row["input_tokens"] for row in curve)), ("padded_tokens", sum(row["padded_tokens"] for row in curve)),
                                ("planned_updates", len(schedule))):
        audit.equal(report["training"][key], expected_value, "training report " + key)
    if report["training"]["stop_reason"] == "scheduled_pass_complete":
        audit.equal(len(curve), len(schedule), "complete pass actually finished")


def audit_benchmark(audit, value, rows, policy, report, config):
    mode = value["mode"]
    chosen, counts = [], Counter()
    for row in rows:
        if counts[row["is_impossible"]] < config["benchmark_per_answerability"]:
            chosen.append(row)
            counts[row["is_impossible"]] += 1
    by_id = {row["id"]: row for row in chosen}
    arms = ("generator", "reader") if mode == "paired" else (mode,)
    repeats = config["benchmark_repetitions"] if mode == "paired" else 1
    expected_order = []
    for repeat in range(repeats):
        for index, row in enumerate(chosen):
            order = (("generator", "reader") if (repeat + index) % 2 == 0 else ("reader", "generator")) if mode == "paired" else arms
            expected_order.extend([row["id"], repeat, arm, position] for position, arm in enumerate(order))
    records = value["records"]
    audit.equal([[r["id"], r["repetition"], r["arm"], r["order_within_pair"]] for r in records], expected_order, "fixed benchmark cohorts and alternating order")
    expected_outputs = {"generator": {r["id"]: r for r in report["test"]["generator"]["records"]},
                        "reader": {r["id"]: r for r in report["test"]["diagnostic_candidate"]["records"]}}
    for record in records + value["warmups"]:
        row = by_id[record["id"]]
        for key in ("context_sha256", "question_sha256", "is_impossible"):
            audit.equal(record[key], row[key], "benchmark full input identity")
        audit.equal(record["movie_sha256"], row["article_sha256"], "benchmark movie identity")
        audit.equal(record["matches_saved_batched_output"], record["prediction_sha256"] == expected_outputs[record["arm"]][record["id"]]["prediction_sha256"], "benchmark raw replay check")
        audit.check(record["seconds"] > 0 and math.isfinite(record["seconds"]), "positive benchmark duration")
    pairs = []
    if mode == "paired":
        for start in range(0, len(records), 2):
            pair = {row["arm"]: row for row in records[start:start + 2]}
            first, second = pair["generator"], pair["reader"]
            pairs.append({"id": first["id"], "repetition": first["repetition"], "generator_seconds": first["seconds"], "reader_seconds": second["seconds"],
                          "same_complete_input": first["context_sha256"] == second["context_sha256"] and first["question_sha256"] == second["question_sha256"],
                          "generator_over_reader_speed_ratio": first["seconds"] / second["seconds"]})
    audit.equal(value["pairs"], pairs, "paired input and latency arithmetic")
    audit.equal(value["median_paired_speed_ratio"], statistics.median(row["generator_over_reader_speed_ratio"] for row in pairs) if pairs else None, "paired median speed ratio")
    audit.equal(value["all_paired_inputs_identical"], all(row["same_complete_input"] for row in pairs), "matched complete benchmark inputs")
    audit.equal(value["replay_matches"], {arm: sum(row["matches_saved_batched_output"] for row in records if row["arm"] == arm) for arm in arms}, "benchmark replay totals")
    audit.equal(value["by_arm"], {arm: {"requests": len(group), "median_seconds": statistics.median(row["seconds"] for row in group), "total_seconds": sum(row["seconds"] for row in group)} for arm in arms if (group := [row for row in records if row["arm"] == arm])}, "benchmark arm totals")
    audit.equal(value["candidate_checkpoint"], policy["diagnostic_checkpoint"], "benchmark checkpoint binding")
    audit.equal(value["candidate_threshold"], policy["diagnostic_threshold"], "benchmark threshold binding")
    audit.equal(value["selected_deployment_kind"], policy["kind"], "benchmark deployment disclosure")
    audit.equal(value["peak_process_rss_gib"], value["peak_process_rss_bytes"] / 1024**3, "benchmark RSS units")


def privacy(audit, value):
    forbidden = {"context", "question", "prediction", "span_prediction", "references", "response", "movie_id", "article", "answer", "input_ids", "raw_logits"}
    if isinstance(value, dict):
        audit.check(not (set(value) & forbidden), "public artifact contains no raw text/weights fields")
        for child in value.values():
            privacy(audit, child)
    elif isinstance(value, list):
        for child in value:
            privacy(audit, child)
    elif isinstance(value, str):
        audit.check(not re.search(r"(?:^|[\s\"'])[A-Za-z]:[\\/]|(?:^|\s)/(?:Users|home|tmp)/|\\\\[^\s\\]+\\", value), "public artifact contains no private absolute path")


def run(args):
    audit = Audit()
    directory = args.run
    audit.check(read(directory / "run_status.json")["status"] == "complete", "completed reader run")
    report = read(directory / "report.json")
    audit.equal(set(report["artifact_sha256"]), PUBLIC, "14 public artifact allowlist")
    artifacts = {name: audit.checked_read(directory / name, expected) for name, expected in report["artifact_sha256"].items()}
    config, protocol, policy, selection = [artifacts[name + ".json"] for name in ("resolved_config", "protocol", "policy", "selection")]
    audit.equal(config, protocol["config"], "protocol config")
    audit.equal(config, policy["generation_config"], "policy config")
    audit.equal(set(protocol["software_sha256"]), SOFTWARE, "frozen software manifest")
    for name, expected in protocol["software_sha256"].items():
        audit.equal(digest((ROOT / "src/cpu_ai_lab" / name).read_text(encoding="utf-8")), expected, "frozen source " + name)
    for name in ("selection", "protocol"):
        audit.equal(policy[name + "_sha256"], file_hash(directory / (name + ".json")), "policy binding " + name)
    audit.check(selection["uses_test"] is False and policy["selection_uses_test"] is False and report["selection"]["test_used_for_selection"] is False, "validation-only selection")
    times = [protocol["created_utc"], selection["completed_utc"], policy["created_utc"], report["selection"]["test_started_utc"], report["completed_utc"]]
    audit.check(list(map(datetime.fromisoformat, times)) == sorted(map(datetime.fromisoformat, times)), "protocol, selection, policy, test chronology")
    audit.equal(report["selection"]["policy_saved_utc"], policy["created_utc"], "saved policy timestamp")
    audit.equal(protocol["prompt_instruction"], data.PROMPT_INSTRUCTION, "full-context prompt contract")

    splits, provenance = data.load_prepared(args.prepared, config)
    audit.equal(provenance, report["dataset"], "report prepared data binding")
    audit.equal(provenance, artifacts["data_provenance.json"], "public provenance binding")
    audit.equal(data.public_manifest(splits), artifacts["split_manifest.json"], "public selected data manifest")
    prior = [read(path) for path in args.prior_manifests]
    prior_hashes = [file_hash(path) for path in args.prior_manifests]
    audit.equal(provenance["prior_manifest_sha256"], prior_hashes, "ordered previous manifest binding")
    prior_rows = [row for manifest in prior for name in data.SPLITS for row in manifest[name]]
    old_ids, old_contexts = {row["id"] for row in prior_rows}, {row["context_sha256"] for row in prior_rows}
    controls = provenance["historical_exclusion_controls"]
    audit.equal(controls["all_prior_ids"], sorted(old_ids), "all previous question exclusions")
    audit.equal(controls["all_prior_context_sha256"], sorted(old_contexts), "all previous full-context exclusions")
    selected_by_id = {row["source_question_id"]: row for rows in splits.values() for row in rows}
    source_found = set()
    for name, expected in data.SOURCES.items():
        original = audit.checked_read(args.source_cache / name, expected)
        for movie in original:
            for question in movie["qa"]:
                row = selected_by_id.get(question["id"])
                if row is None:
                    continue
                source_found.add(question["id"])
                for key, expected_value in (("context", movie["plot"]), ("movie_id", movie["id"]), ("article", movie["title"]),
                                             ("question", question["question"]), ("references", question["answers"]), ("is_impossible", question["no_answer"]), ("source", name)):
                    audit.equal(row[key], expected_value, "unchanged native DuoRC " + key)
    audit.equal(source_found, set(selected_by_id), "every selected question has a native source")
    audit.equal(file_hash(args.source_cache / "LICENSE"), data.LICENSE_SHA256, "source license bytes")
    seen = {key: set() for key in ("id", "context_sha256", "movie_sha256")}
    for name, rows in splits.items():
        audit.check(len(rows) == config[name + "_size"] and sum(row["is_impossible"] for row in rows) * 2 == len(rows), name + " balanced answerability")
        audit.check(max(Counter(row["movie_sha256"] for row in rows).values()) <= config["max_questions_per_movie"], name + " movie cap")
        audit.check(len({(row["movie_sha256"], row["question_sha256"]) for row in rows}) == len(rows), name + " distinct movie questions")
        for key in seen:
            current = {row[key] for row in rows}
            audit.check(not current & seen[key], name + " cross-split " + key + " separation")
            seen[key].update(current)
        audit.check(not {row["id"] for row in rows} & old_ids and not {row["source_question_id"] for row in rows} & old_ids and not {row["context_sha256"] for row in rows} & old_contexts, name + " historical exclusions")
        audit.equal(provenance["selection"][name], {"by_stratum": dict(Counter(data.stratum(row) for row in rows)), "movies": len({row["movie_sha256"] for row in rows}),
                    "contexts": len({row["context_sha256"] for row in rows}), "max_questions_per_movie": max(Counter(row["movie_sha256"] for row in rows).values())}, name + " dataset totals")

    initial = audit_checkpoint(audit, directory, "initial", report, policy)
    final = audit_checkpoint(audit, directory, "final", report, policy)
    audit.check(initial["files_sha256"]["model.safetensors"] != final["files_sha256"]["model.safetensors"], "saved weight bytes changed")
    audit.check(report["training"]["weights_changed"] is True and report["training"]["initial_state_sha256"] != report["training"]["final_state_sha256"], "nonzero actual parameter adaptation")
    from tokenizers import Tokenizer
    tokenizer = Tokenizer.from_file(str(directory / "initial_reader/tokenizer.json"))
    encodings = {}
    for rows in splits.values():
        for row in rows:
            encoded = tokenizer.encode(row["question"], row["context"], add_special_tokens=True)
            audit.equal(len(encoded.ids), row["reader_tokens"], "complete reader token count")
            audit.check(len(encoded.ids) <= config["reader_max_length"], "untruncated complete reader input budget")
            starts = {pair[0]: index for index, pair in enumerate(encoded.offsets) if encoded.sequence_ids[index] == 1 and pair[1] > pair[0]}
            ends = {pair[1]: index for index, pair in enumerate(encoded.offsets) if encoded.sequence_ids[index] == 1 and pair[1] > pair[0]}
            for span in row["literal_answer_spans"]:
                audit.check(span["start"] in starts and span["end"] in ends and 0 <= ends[span["end"]] - starts[span["start"]] < config["reader_max_answer_tokens"], "exact bounded literal target boundaries")
                audit.equal(row["context"][span["start"]:span["end"]], span["text"], "literal native target")
                audit.check(span["text"] in row["references"], "literal target belongs to original references")
            encodings[row["id"]] = encoded
    audit_schedule(audit, splits["train"], artifacts["sample_schedule.json"], artifacts["training_curve.json"], report, config)
    audit.check(report["cpu"]["cuda_build"] is None and report["cpu"]["cuda_available"] is False, "CPU-only build")
    audit.equal(report["cpu"]["reader_parameter_devices"], ["cpu"], "reader on CPU")
    audit.equal(report["cpu"]["optimizer_devices"], ["cpu"], "optimizer on CPU")
    audit.equal(report["cpu"]["generator_trainable_parameters"], 0, "frozen generator")
    audit.equal(report["timing"]["peak_process_rss_gib"], report["timing"]["peak_process_rss_bytes"] / 1024**3, "training RSS units")
    audit.check(report["timing"]["full_pipeline_seconds"] >= sum(report["timing"]["stages_seconds"].values()), "stage time fits full duration")

    local = {name: audit.checked_read(directory / name, expected) for name, expected in report["local_prediction_sha256"].items()}
    audit.equal(set(local), {"generator-validation-local.json", "generator-test-local.json", "initial-validation-local.json", "final-validation-local.json", "initial-test-local.json", "final-test-local.json"}, "six local output artifacts")
    baselines = {}
    for name in ("validation", "test"):
        raw = local["generator-" + name + "-local.json"]
        audit_generator(audit, splits[name], raw, artifacts["generator-" + name + ".json"])
        audit.equal(report[name]["generator"], artifacts["generator-" + name + ".json"], "report generator metric binding")
        baselines[name] = baseline_metric(splits[name], [row["prediction"] for row in raw])
        for stage in ("initial", "final"):
            stem = stage + "-" + name
            audit_decoded(audit, splits[name], local[stem + "-local.json"], artifacts[stem + "-decoded.json"], encodings, config)
    for stage in ("initial", "final"):
        decoded = local[stage + "-validation-local.json"]
        audit_thresholds(audit, selection["checkpoints"][stage], splits["validation"], decoded, baselines["validation"])
        audit.equal(report["validation"][stage + "_fixed_zero"], reader_metric(splits["validation"], decoded, 0.0, baselines["validation"]), stage + " zero-threshold validation")
        chosen = selection["checkpoints"][stage]
        decoded = local[stage + "-test-local.json"]
        expected = {"fixed_zero": reader_metric(splits["test"], decoded, 0.0, baselines["test"]),
                    "diagnostic": reader_metric(splits["test"], decoded, chosen["diagnostic_threshold"], baselines["test"]),
                    "threshold_selected": reader_metric(splits["test"], decoded, chosen["selected_threshold"], baselines["test"]) if chosen["accepted"] else None}
        audit.equal(report["test"][stage], expected, stage + " every held-out score")
    choice = choose_checkpoint(selection["checkpoints"])
    audit.equal(selection["choice"], choice, "independent two-checkpoint selection")
    for key, value in choice.items():
        audit.equal(policy[key], value, "frozen policy " + key)
        audit.equal(report["selection"][key], value, "report choice " + key)
    audit.equal(report["test"]["diagnostic_candidate"], reader_metric(splits["test"], local[choice["diagnostic_checkpoint"] + "-test-local.json"], choice["diagnostic_threshold"], baselines["test"]), "chosen diagnostic test scores")
    audit.equal(report["test"]["selected"], report["test"][choice["checkpoint"]]["threshold_selected"] if choice["kind"] == "reader" else report["test"]["generator"], "actual deployment test scores")

    source = read(args.source / "metrics.json")
    audit.equal(protocol["generator_source_metrics_sha256"], file_hash(args.source / "metrics.json"), "generator source metrics")
    audit.equal(policy["generator_source"], report["generator_source"], "generator source policy binding")
    audit.equal(report["generator_source"]["split_manifest_sha256"], prior_hashes[0], "generator source historical exclusion")
    audit.equal(source["dataset"]["split_manifest_sha256"], prior_hashes[0], "original generator manifest")
    for name, expected in source["adapter_sha256"].items():
        audit.equal(file_hash(args.source / "adapter" / name), expected, "original adapter bytes")
    audit.equal(report["reader_source"], protocol["reader_source"], "reader initializer source")
    audit.equal(initial["initializer"]["weights_sha256"], config["reader_weights_sha256"], "pinned reader initializer")
    initializer = args.model_cache / ("models--" + config["reader_model_id"].replace("/", "--")) / "snapshots" / config["reader_revision"] / "model.safetensors"
    audit.equal(file_hash(initializer), config["reader_weights_sha256"], "actual pinned initializer weight bytes")

    benchmark_hashes = {}
    for name in sorted(BENCHMARKS):
        path = directory / name
        if not path.exists():
            continue
        value = read(path)
        audit.equal(value["source_report_sha256"], file_hash(directory / "report.json"), "benchmark source report")
        audit.equal(value["policy_sha256"], file_hash(directory / "policy.json"), "benchmark frozen policy")
        audit_benchmark(audit, value, splits["test"], policy, report, config)
        benchmark_hashes[name] = file_hash(path)
    public_hashes = {}
    if args.public is not None:
        names = {path.name for path in args.public.iterdir()}
        if args.output is not None and args.output.resolve().parent == args.public.resolve():
            names.discard(args.output.name)
        expected = PUBLIC | {"report.json"} | set(benchmark_hashes)
        if "analysis.json" in names:
            expected.add("analysis.json")
            analysis = read(args.public / "analysis.json")
            audit.equal(analysis["source_report_sha256"], file_hash(directory / "report.json"), "supplemental analysis binding")
        audit.equal(names, expected, "public exact file allowlist")
        for name in sorted(names):
            if name != "analysis.json":
                audit.equal(file_hash(args.public / name), file_hash(directory / name), "identical public bytes " + name)
            privacy(audit, read(args.public / name))
            public_hashes[name] = file_hash(args.public / name)
    else:
        for value in [report, *artifacts.values(), *[read(directory / name) for name in benchmark_hashes]]:
            privacy(audit, value)
    audit.check("torch" not in sys.modules and "transformers" not in sys.modules, "audit never imported a model runtime")
    return {"format": "cpu-reader-audit-v1", "passed": True, "checks": audit.checks,
            "counts": {name: len(rows) for name, rows in splits.items()}, "completed_updates": len(artifacts["training_curve.json"]),
            "choice": choice, "historical_exclusions": {"ids": len(old_ids), "contexts": len(old_contexts)},
            "report_sha256": file_hash(directory / "report.json"), "audit_script_sha256": digest(Path(__file__).read_text(encoding="utf-8")),
            "benchmark_sha256": benchmark_hashes, "public_sha256": public_hashes,
            "limitations": ["No model inference or training is repeated; saved best-span/null scores cannot prove the maximizing decoder choice without full logits.",
                            "Tensor bytes prove that saved weights changed; CPU execution is supported by recorded runtime/device evidence and frozen code, not remote hardware attestation.",
                            "Generator token counts and span-eligibility limits are hash-bound preparation metadata; reader tokenizer lengths/alignment are independently recomputed.",
                            "Saved timestamps support policy-before-test ordering but are not cryptographic proof of chronology."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/reader-run")
    parser.add_argument("--prepared", type=Path, default=ROOT / "artifacts/reader-data")
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/document-task-v1")
    parser.add_argument("--source-cache", type=Path, default=ROOT / ".cache/duorc-research")
    parser.add_argument("--model-cache", type=Path, default=ROOT / ".cache/huggingface/hub")
    parser.add_argument("--prior-manifests", type=Path, nargs="+", default=[ROOT / "results/document-split-manifest.json", ROOT / "results/document-v2-split-manifest.json", ROOT / "results/document-verifier/split_manifest.json"])
    parser.add_argument("--public", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = run(args)
    payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if args.output:
        if args.output.exists() and args.output.read_bytes() != payload:
            raise FileExistsError("Existing audit output is preserved; choose a new output file.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    print(payload.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
