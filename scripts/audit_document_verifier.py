"""Replay a completed verifier experiment without model inference or Torch.

Run only when timed CPU experiments have finished. Raw generations and prepared
data stay local; the optional audit output contains counts and checksums only.
This checks saved confidence statistics, not the unavailable token logits that
produced them. It replays the small logistic fit, never the language model.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import statistics
import string
import sys

# Keep the optional classifier replay small and independent of benchmark threads.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from cpu_ai_lab import document_v2_metrics as scoring
from cpu_ai_lab import document_verifier as verifier

SPLITS = ("train", "validation", "test")
STRATA = ("answerable_short", "answerable_long", "missing_short", "missing_long")
PUBLIC_FILES = {
    "resolved_config.json", "protocol.json", "selection.json", "policy.json",
    "verifier.json", "data_provenance.json", "split_manifest.json",
    "train-features.json", "validation-features.json", "test-features.json",
}
BENCHMARK_FILES = {"benchmark-paired.json", "benchmark-baseline.json", "benchmark-verifier.json"}
MANIFEST_FIELDS = (
    "id", "source", "article_sha256", "context_sha256", "question_sha256",
    "is_impossible", "length_group", "prompt_tokens", "context_tokens", "reference_token_counts",
)


def read(path):
    return json.loads(Path(path).read_bytes())


def digest(value):
    return hashlib.sha256(value.encode("utf-8") if isinstance(value, str) else value).hexdigest()


def file_hash(path):
    return digest(Path(path).read_bytes())


def normalize(value):
    value = "".join(char for char in value.lower() if char not in string.punctuation)
    return " ".join(re.sub(r"\b(a|an|the)\b", " ", value).split())


def independent_features(context, question, prediction, confidence):
    """Separate formula implementation; this function cannot receive gold data."""
    passage, query, answer = map(normalize, (context, question, prediction))
    passage_words, query_words, answer_words = passage.split(), query.split(), answer.split()
    passage_set, query_set, answer_set = set(passage_words), set(query_words), set(answer_words)
    overlap = lambda first, second: len(first.intersection(second)) / len(first) if first else 0.0
    start = passage.find(answer) if answer else -1
    answer_numbers, query_numbers = set(re.findall(r"\d+", answer)), set(re.findall(r"\d+", query))
    return [float(confidence[name]) for name in verifier.FEATURE_NAMES[:6]] + [
        float(start >= 0), float(len(answer_words)), float(len(answer)),
        len(answer_set) / len(answer_words) if answer_words else 0.0,
        overlap(answer_set, passage_set), overlap(answer_set, query_set), overlap(query_set, passage_set),
        float(len(passage_words)), float(len(query_words)), start / len(passage) if start >= 0 and passage else -1.0,
        float(bool(answer_numbers)), float(bool(answer_numbers & query_numbers)),
        float(bool(query_words) and query_words[0] == "who"),
        float(bool(query_words) and query_words[0] == "when"),
    ]


class Audit:
    def __init__(self):
        self.checks = 0

    def check(self, condition, label):
        self.checks += 1
        if not condition:
            raise ValueError("Audit failed: " + label)

    def equal(self, actual, expected, label, tolerance=1e-10):
        """Compare JSON recursively without printing raw data on failure."""
        if isinstance(expected, dict):
            self.check(isinstance(actual, dict) and set(actual) == set(expected), label + " keys")
            for key in expected:
                self.equal(actual[key], expected[key], label + "/" + key, tolerance)
        elif isinstance(expected, list):
            self.check(isinstance(actual, list) and len(actual) == len(expected), label + " length")
            for index, (first, second) in enumerate(zip(actual, expected, strict=True)):
                self.equal(first, second, label + "/" + str(index), tolerance)
        elif type(expected) in (int, float) and type(actual) in (int, float):
            self.check(math.isclose(actual, expected, rel_tol=tolerance, abs_tol=tolerance), label)
        else:
            self.check(type(actual) is type(expected) and actual == expected, label)

    def checked_read(self, path, expected):
        self.check(file_hash(path) == expected, Path(path).name + " checksum")
        return read(path)


def public_rows(rows):
    return [{**{key: row[key] for key in MANIFEST_FIELDS},
             "reference_sha256": [digest(value) for value in row["references"]],
             "response_sha256": digest(row["response"])} for row in rows]


def group(row):
    return ("missing_" if row["is_impossible"] else "answerable_") + row["length_group"]


def plain_metric(rows, predictions):
    records = [scoring.score_prediction(row, prediction) for row, prediction in zip(rows, predictions, strict=True)]
    return {**scoring.summarize(records), "records": records}


def independent_gate(baseline, candidate):
    correct = round(baseline["answerable_exact_match"] * baseline["answerable_examples"])
    retained = round(candidate["answerable_exact_match"] * candidate["answerable_examples"])
    checks = {
        "positive_baseline_exact_correct_count": correct > 0,
        "exact_correct_retention_at_least_0_95": correct > 0 and retained >= 0.95 * correct,
        "answerable_f1_drop_at_most_0_02": candidate["answerable_token_f1"] >= baseline["answerable_token_f1"] - 0.02,
        "missing_recall_at_least_0_40": candidate["correct_abstention_recall"] >= 0.4,
    }
    return {"accepted": all(checks.values()), "checks": checks,
            "baseline_exact_correct": correct, "retained_exact_correct": retained,
            "exact_correct_retention": retained / correct if correct else None,
            "retention_and_f1_pass": all(value for key, value in checks.items() if key != "missing_recall_at_least_0_40")}


def audit_selection(audit, stored, rows, predictions, probabilities):
    baseline = verifier.evaluate_policy(rows, predictions, probabilities, None, enabled=False)
    audit.equal(stored["baseline"], baseline, "validation baseline")
    grid = [index / 100 for index in range(101)] + ["always_abstain"]
    audit.equal(stored["criteria"]["threshold_grid"], grid, "threshold grid")
    audit.check(len(stored["trials"]) == len(grid), "one trial per fixed threshold")
    ranked, constrained = [], []
    for index, (threshold, trial) in enumerate(zip(grid, stored["trials"], strict=True)):
        metric = verifier.evaluate_policy(rows, predictions, probabilities, threshold)
        gate = independent_gate(baseline, metric)
        audit.equal(trial, {"threshold": threshold, "metrics": {key: value for key, value in metric.items() if key != "records"}, "gate": gate}, "validation trial")
        key = (metric["correct_abstention_recall"], metric["answerable_token_f1"], metric["answerable_exact_match"], metric["coverage"], -index)
        if gate["accepted"]:
            ranked.append((key, trial, metric))
        if gate["retention_and_f1_pass"]:
            constrained.append((key, trial))
    best = max(ranked, key=lambda value: value[0]) if ranked else None
    best_constrained = max(constrained, key=lambda value: value[0])[1] if constrained else None
    audit.equal(stored["best_constrained_candidate"], best_constrained, "best constrained validation candidate")
    audit.equal(stored["selected_enabled"], bool(best), "selected policy enabled")
    audit.equal(stored["accepted"], bool(best), "validation acceptance")
    audit.equal(stored["selected_threshold"], best[1]["threshold"] if best else None, "selected threshold")
    audit.equal(stored["selected_gate"], best[1]["gate"] if best else None, "selected gate")
    audit.equal(stored["selected"], best[2] if best else baseline, "selected validation metrics")
    audit.equal(stored["validation_identity_sha256"], baseline["evaluation_identity_sha256"], "validation identity")
    audit.check(stored["test_used_for_selection"] is False, "threshold selection excludes test")


def candidate_threshold(selection):
    if selection["accepted"]:
        return selection["selected_threshold"]
    trial = selection["best_constrained_candidate"]
    return trial["threshold"] if trial else 0.0


def audit_benchmark(audit, benchmark, report, policy, local, rows, feature_rows):
    mode = benchmark["mode"]
    audit.equal(benchmark["selected_deployment_verifier_enabled"], policy["enabled"], "benchmark deployment status")
    audit.check(benchmark["candidate_enabled_for_cost_measurement"] is True, "benchmark measures candidate cost")
    selected, counts = [], Counter()
    for row in rows:
        if counts[group(row)] < policy["generation_config"]["benchmark_per_stratum"]:
            selected.append(row["id"])
            counts[group(row)] += 1
    repetitions = policy["generation_config"]["benchmark_repetitions"] if mode == "paired" else 1
    expected_raw = {row["id"]: row for row in local}
    candidate = {row["id"]: row for row in report["test"]["verifier_candidate"]["records"]}
    features = {row["id"]: row for row in feature_rows}
    records = benchmark["records"]
    audit.check(len(records) == len(selected) * repetitions * (2 if mode == "paired" else 1), "benchmark request count")
    expected_schedule = []
    for repeat in range(repetitions):
        for index, identifier in enumerate(selected):
            arms = (("baseline", "verifier") if (repeat + index) % 2 == 0 else ("verifier", "baseline")) if mode == "paired" else (mode,)
            expected_schedule.extend((identifier, repeat, arm, offset) for offset, arm in enumerate(arms))
    audit.equal([[r["id"], r["repetition"], r["arm"], r["order_within_pair"]] for r in records], [list(value) for value in expected_schedule], "benchmark schedule")
    for record in records + benchmark["warmups"]:
        identifier = record["id"]
        audit.check(identifier in selected and record["seconds"] > 0, "benchmark selected record and positive duration")
        audit.equal(record["matches_saved_batched_generator"], record["raw_prediction_sha256"] == expected_raw[identifier]["prediction_sha256"], "benchmark raw reload comparison")
        if record["arm"] == "verifier":
            audit.equal(record["matches_saved_batched_candidate"], record["prediction_sha256"] == candidate[identifier]["prediction_sha256"], "benchmark candidate reload comparison")
            audit.equal(record["probability_delta_from_saved_batch"], abs(record["verifier_probability"] - features[identifier]["verifier_probability"]), "benchmark probability delta")
    by_arm = {arm: {"requests": len(values), "median_seconds": statistics.median(r["seconds"] for r in values), "total_seconds": sum(r["seconds"] for r in values)} for arm in ("baseline", "verifier") if (values := [r for r in records if r["arm"] == arm])}
    audit.equal(benchmark["by_arm"], by_arm, "benchmark latency totals")
    expected_pairs = []
    if mode == "paired":
        for offset in range(0, len(records), 2):
            pair = {r["arm"]: r for r in records[offset:offset + 2]}
            baseline, verified = pair["baseline"], pair["verifier"]
            matched = (baseline["raw_prediction_sha256"] == verified["raw_prediction_sha256"] and baseline["generated_tokens_including_eos"] == verified["generated_tokens_including_eos"])
            expected_pairs.append({"id": baseline["id"], "repetition": baseline["repetition"], "matched_work": matched,
                                   "baseline_seconds": baseline["seconds"], "verifier_seconds": verified["seconds"],
                                   "verifier_over_baseline_latency_ratio": verified["seconds"] / baseline["seconds"] if matched else None})
    audit.equal(benchmark["pairs"], expected_pairs, "benchmark matched pairs")
    matched = [row for row in expected_pairs if row["matched_work"]]
    audit.equal(benchmark["matched_pairs"], len(matched), "benchmark matched pair count")
    audit.equal(benchmark["total_pairs"], len(expected_pairs), "benchmark total pair count")
    audit.equal(benchmark["median_paired_latency_ratio"], statistics.median(row["verifier_over_baseline_latency_ratio"] for row in matched) if matched else None, "benchmark paired median")
    audit.equal(benchmark["raw_generator_reload_matches"], sum(r["matches_saved_batched_generator"] for r in records), "benchmark raw matches")
    audit.equal(benchmark["candidate_reload_matches"], sum(r["matches_saved_batched_candidate"] is True for r in records), "benchmark candidate matches")
    audit.equal(benchmark["peak_process_rss_gib"], benchmark["peak_process_rss_bytes"] / 1024**3, "benchmark RSS units")


def audit_public_privacy(audit, value):
    forbidden = {"context", "question", "prediction", "response", "references", "prompt", "article", "answer", "raw_logits", "input_ids"}
    if isinstance(value, dict):
        audit.check(not (set(value) & forbidden), "no raw dataset/generation fields in public JSON")
        for child in value.values():
            audit_public_privacy(audit, child)
    elif isinstance(value, list):
        for child in value:
            audit_public_privacy(audit, child)
    elif isinstance(value, str):
        audit.check(not re.search(r"(?:^|[\s\"'])[A-Za-z]:[\\/]|(?:^|\s)/(?:Users|home|tmp)/|\\\\[^\s\\]+\\", value), "no private absolute paths in public JSON")


def run(args):
    audit = Audit()
    directory, prepared = args.run, args.prepared
    audit.check(read(directory / "run_status.json")["status"] == "complete", "completed experiment")
    report = read(directory / "report.json")
    audit.equal(set(report["artifact_sha256"]), PUBLIC_FILES, "complete public artifact allowlist")
    artifacts = {name: audit.checked_read(directory / name, expected) for name, expected in report["artifact_sha256"].items()}
    config, protocol, policy, selection, classifier, provenance, manifest = [artifacts[name + ".json"] for name in ("resolved_config", "protocol", "policy", "selection", "verifier", "data_provenance", "split_manifest")]
    audit.equal(provenance, report["dataset"], "report dataset binding")
    audit.equal(provenance, read(prepared / "data_provenance.json"), "prepared provenance binding")
    audit.equal(file_hash(directory / "split_manifest.json"), provenance["split_manifest_sha256"], "prepared manifest binding")
    audit.equal(file_hash(prepared / "split_manifest.json"), provenance["split_manifest_sha256"], "local manifest binding")
    audit.equal(provenance["data_config"], {key: config[key] for key in provenance["data_config"]}, "data configuration binding")
    audit.equal(config, protocol["config"], "protocol configuration")
    audit.equal(config, policy["generation_config"], "policy configuration")
    for key, expected in (("verifier_l2", 0.01), ("verifier_max_iterations", 100),
                          ("minimum_correct_answer_retention", 0.95), ("minimum_missing_answer_recall", 0.4),
                          ("maximum_answerable_f1_loss", 0.02), ("threshold_grid_step", 0.01)):
        audit.equal(config[key], expected, "fixed protocol " + key)
    for name in ("verifier", "selection", "protocol"):
        audit.equal(policy[name + "_sha256"], file_hash(directory / (name + ".json")), "policy " + name + " binding")
    audit.equal(set(protocol["software_sha256"]), {"document_verifier_run.py", "document_verifier.py", "document_verifier_data.py", "document_v2_data.py", "document_v2_metrics.py"}, "frozen software manifest")
    for name, expected in protocol["software_sha256"].items():
        audit.check(Path(name).name == name, "software file is a basename")
        audit.equal(digest((ROOT / "src" / "cpu_ai_lab" / name).read_text(encoding="utf-8")), expected, "frozen software " + name)
    audit.equal(protocol["feature_names"], list(verifier.FEATURE_NAMES), "feature contract")
    audit.check(policy["selection_uses_test"] is False and selection["uses_test"] is False and report["selection"]["test_used_for_selection"] is False, "validation-only selection declarations")
    times = [protocol["created_utc"], selection["completed_utc"], policy["created_utc"], report["selection"]["test_generation_started_utc"], report["completed_utc"]]
    audit.check(list(map(datetime.fromisoformat, times)) == sorted(map(datetime.fromisoformat, times)), "protocol then selection then policy then test then report chronology")
    audit.equal(report["selection"]["policy_saved_utc"], policy["created_utc"], "policy timestamp binding")

    prior = [read(path) for path in args.prior_manifests]
    prior_hashes = [file_hash(path) for path in args.prior_manifests]
    controls = provenance["historical_exclusion_controls"]
    audit.equal(provenance["prior_manifest_sha256"], prior_hashes, "all ordered historical manifest checksums")
    prior_rows = [row for previous in prior for name in SPLITS for row in previous[name]]
    old_ids = {row["id"] for row in prior_rows}
    old_contexts = {row["context_sha256"] for row in prior_rows}
    old_articles = {row["article_sha256"] for row in prior_rows}
    ancestor_articles = {row["article_sha256"] for name in ("train", "validation") for row in prior[0][name]}
    for key, values in (("all_prior_ids", old_ids), ("all_prior_context_sha256", old_contexts), ("all_prior_article_sha256", old_articles), ("ancestor_train_and_validation_article_sha256", ancestor_articles)):
        audit.equal(controls[key], sorted(values), key)
    audit.equal(controls["prior_manifests"], [{"index": index, "sha256": prior_hashes[index], "split_counts": {name: len(previous[name]) for name in SPLITS}} for index, previous in enumerate(prior)], "exclusion manifest bindings")
    audit.equal(controls["original_article_holdout_manifest_sha256"], prior_hashes[0], "original article holdout binding")
    audit.equal(controls["prior_record_count_with_repetition"], len(prior_rows), "historical record total")

    splits, local, matrices, probabilities = {}, {}, {}, {}
    seen_ids, seen_contexts, seen_articles = set(), set(), set()
    for name in SPLITS:
        payload = (prepared / "data" / (name + ".jsonl")).read_bytes()
        audit.equal(digest(payload), provenance["local_data_sha256"][name], name + " local data checksum")
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
        splits[name] = rows
        audit.equal(public_rows(rows), manifest[name], name + " manifest projection")
        audit.equal(len(rows), config[name + "_size"], name + " configured size")
        audit.equal(dict(Counter(group(row) for row in rows)), {key: len(rows) // 4 for key in STRATA}, name + " four balanced strata")
        ids, contexts, articles = ({row[key] for row in rows} for key in ("id", "context_sha256", "article_sha256"))
        audit.check(len(ids) == len(rows) == len(contexts), name + " one question per unique context")
        audit.check(not (ids & (old_ids | seen_ids)) and not (contexts & (old_contexts | seen_contexts)), name + " fresh IDs and complete passages")
        audit.check(not (articles & seen_articles), name + " current article separation")
        if name == "train":
            audit.check(articles <= ancestor_articles, "training uses the original adaptation article pool")
        else:
            audit.check(not (articles & ancestor_articles), name + " original adaptation article holdout")
        seen_ids.update(ids)
        seen_contexts.update(contexts)
        seen_articles.update(articles)
        audit.equal(provenance["selection"]["historical_article_reuse"][name], {"selected_articles_seen_in_any_prior_manifest": len(articles & old_articles), "examples_on_previously_used_articles": sum(row["article_sha256"] in old_articles for row in rows)}, name + " historical topic reuse disclosure")
        for row in rows:
            audit.check(type(row["is_impossible"]) is bool, name + " native boolean label")
            audit.equal(row["prompt"], protocol["prompt_instruction"] + row["context"] + "\nQuestion: " + row["question"], name + " gold-free complete prompt")
            for key in ("article", "context", "question"):
                audit.equal(row[key + "_sha256"], digest(row[key]), name + " " + key + " integrity")
            audit.check(0 < row["context_tokens"] <= config["max_context_tokens"] and row["prompt_tokens"] <= config["max_prompt_tokens"], name + " recorded token limits")
            audit.equal(row["length_group"], "short" if row["context_tokens"] <= config["short_context_tokens"] else "long", name + " length classification")
            audit.check(all(0 < count <= config["max_answer_tokens"] for count in row["reference_token_counts"]), name + " answer token limits")
            audit.equal(row["response"], "NOT_FOUND" if row["is_impossible"] else row["references"][0], name + " response target")
            audit.check((not row["references"]) == row["is_impossible"], name + " native references")
        local[name] = audit.checked_read(directory / (name + "-local.json"), report["local_prediction_sha256"][name + "-local.json"])
        published = artifacts[name + "-features.json"]
        audit.equal([row["id"] for row in local[name]], [row["id"] for row in rows], name + " generation order")
        audit.equal([row["id"] for row in published], [row["id"] for row in rows], name + " public feature order")
        rebuilt = []
        for row, generated, public in zip(rows, local[name], published, strict=True):
            values = independent_features(row["context"], row["question"], generated["prediction"], generated["confidence"])
            audit.equal(generated["features"], values, name + " independently rebuilt gold-free features")
            audit.equal(public["features"], values, name + " public features")
            audit.equal(generated["prediction_sha256"], digest(generated["prediction"]), name + " raw generation checksum")
            audit.equal(public["prediction_sha256"], generated["prediction_sha256"], name + " public raw checksum")
            audit.equal(public["confidence"], generated["confidence"], name + " public confidence")
            audit.equal(generated["generated_tokens_including_eos"], generated["confidence"]["generated_tokens"], name + " confidence token count")
            audit.equal(generated["ended_with_eos"], generated["confidence"]["ended_with_eos"], name + " confidence EOS")
            rebuilt.append(values)
        matrices[name] = np.asarray(rebuilt, dtype=np.float64)

    # Gold is introduced here, after all features have been rebuilt independently.
    labels = [int(not row["is_impossible"] and normalize(generated["prediction"]) in {normalize(value) for value in row["references"]}) for row, generated in zip(splits["train"], local["train"], strict=True)]
    audit.equal(report["training"]["positive_correct_answers"], sum(labels), "train-only correctness labels")
    audit.equal(report["training"]["negative_answers"], len(labels) - sum(labels), "train-only negative labels")
    audit.equal(report["training"]["examples"], len(labels), "classifier training count")
    audit.equal(report["training"]["classifier_parameters"], len(verifier.FEATURE_NAMES) + 1, "classifier parameter count")
    audit.equal(report["training"]["classifier_reload_max_probability_difference"], 0.0, "classifier reload equality")
    train_metric = plain_metric(splits["train"], [row["prediction"] for row in local["train"]])
    audit.equal(report["training"]["train_baseline"], {key: value for key, value in train_metric.items() if key != "records"}, "training scores")
    expected_mean, expected_scale = matrices["train"].mean(axis=0), matrices["train"].std(axis=0)
    expected_scale[expected_scale <= 1e-12] = 1.0
    audit.equal(classifier["scaler"]["mean"], expected_mean.tolist(), "train-only scaler means")
    audit.equal(classifier["scaler"]["scale"], expected_scale.tolist(), "train-only scaler standard deviations")
    verifier.validate_model(classifier)
    replay = verifier.fit(matrices["train"], labels, l2=config["verifier_l2"], max_iter=config["verifier_max_iterations"])
    for key in ("scaler", "intercept", "coefficients", "training"):
        audit.equal(classifier[key], replay[key], "train-only classifier replay " + key, tolerance=1e-8)
    for name in SPLITS:
        logits = ((matrices[name] - expected_mean) / expected_scale) @ np.asarray(classifier["coefficients"]) + classifier["intercept"]
        probabilities[name] = np.exp(-np.logaddexp(0.0, -logits)).tolist()
        audit.equal([row["verifier_probability"] for row in artifacts[name + "-features.json"]], probabilities[name], name + " classifier probabilities")

    for kind, scores in (("verifier", probabilities["validation"]), ("confidence", [math.exp(row["confidence"]["mean_logprob"]) for row in local["validation"]])):
        audit_selection(audit, selection[kind], splits["validation"], [row["prediction"] for row in local["validation"]], scores)
    audit.equal(policy["enabled"], selection["verifier"]["selected_enabled"], "policy enabled binding")
    audit.equal(policy["threshold"], selection["verifier"]["selected_threshold"], "policy threshold binding")
    audit.equal(policy["diagnostic_candidate_threshold"], candidate_threshold(selection["verifier"]), "diagnostic threshold binding")
    for key in ("enabled", "threshold", "diagnostic_candidate_threshold"):
        audit.equal(report["selection"][key], policy[key], "report policy " + key)
    audit.equal(report["selection"]["accepted"], selection["verifier"]["accepted"], "report acceptance")
    for name in ("validation", "test"):
        rows, generated = splits[name], local[name]
        predictions = [row["prediction"] for row in generated]
        audit.equal(report[name]["baseline"], plain_metric(rows, predictions), name + " raw metrics")
        for prefix, scores, selected_key, candidate_key in (("verifier", probabilities[name], "selected", "verifier_candidate"), ("confidence", [math.exp(row["confidence"]["mean_logprob"]) for row in generated], "confidence_selected", "confidence_candidate")):
            chosen = selection[prefix]
            audit.equal(report[name][candidate_key], verifier.evaluate_policy(rows, predictions, scores, candidate_threshold(chosen)), name + " candidate metrics")
            audit.equal(report[name][selected_key], verifier.evaluate_policy(rows, predictions, scores, chosen["selected_threshold"], enabled=chosen["selected_enabled"]), name + " selected metrics")
        guarded = [prediction.strip() if prediction.strip() == "NOT_FOUND" or (normalize(prediction) and normalize(prediction) in normalize(row["context"])) else "NOT_FOUND" for row, prediction in zip(rows, predictions, strict=True)]
        audit.equal(report[name]["span_guard"], plain_metric(rows, guarded), name + " independent span comparator")

    source = read(args.source / "metrics.json")
    audit.equal(report["source"], policy["source"], "report and policy generator binding")
    audit.equal(report["source"], protocol["source"], "report and protocol generator binding")
    audit.equal(report["source"]["metrics_sha256"], file_hash(args.source / "metrics.json"), "original adapter metrics binding")
    audit.equal(report["source"]["split_manifest_sha256"], prior_hashes[0], "original adapter exclusion binding")
    audit.equal(source["dataset"]["split_manifest_sha256"], prior_hashes[0], "original adapter data binding")
    audit.equal(report["source"]["selected_step"], source["selection"]["step"], "original selected adapter step")
    audit.equal(report["source"]["adapter_sha256"], source["adapter_sha256"], "original adapter file manifest")
    for folder, key in (("adapter", "adapter_sha256"), ("tokenizer", "tokenizer_sha256")):
        audit.equal({path.name: file_hash(path) for path in (args.source / folder).iterdir() if path.is_file()}, source[key], "original " + folder + " bytes")
    audit.check(report["cpu"]["cuda_build"] is None and report["cpu"]["cuda_available"] is False and report["cpu"]["parameter_devices"] == ["cpu"], "CPU-only evidence")
    audit.check(report["training"]["generator_trainable_parameters"] == protocol["generator_trainable_parameters"] == 0, "generator frozen")
    audit.equal(report["cpu"]["initial_adapter_sha256"], report["cpu"]["final_adapter_sha256"], "generator unchanged")
    audit.equal(report["timing"]["peak_process_rss_gib"], report["timing"]["peak_process_rss_bytes"] / 1024**3, "run RSS units")
    audit.check(report["timing"]["seconds"] >= sum(report["timing"]["stages_seconds"].values()), "stage durations bounded by pipeline duration")

    # Match local rows directly to pinned native SQuAD2: no cropping or synthetic negatives.
    by_id = {row["id"]: row for rows in splits.values() for row in rows}
    source_ids = set()
    for filename, expected in provenance["source_sha256"].items():
        original = audit.checked_read(args.source_cache / filename, expected)
        for article in original["data"]:
            for paragraph in article["paragraphs"]:
                for question in paragraph["qas"]:
                    row = by_id.get(question["id"])
                    if row is None:
                        continue
                    source_ids.add(row["id"])
                    audit.equal(row["article"], article["title"], "native source article")
                    audit.equal(row["context"], paragraph["context"], "complete native source context")
                    audit.equal(row["question"], question["question"], "native source question")
                    audit.equal(row["is_impossible"], question["is_impossible"], "native missing-answer label")
                    audit.equal(row["references"], list(dict.fromkeys(answer["text"] for answer in question["answers"])), "native references only")
    audit.equal(source_ids, set(by_id), "all current records present in pinned source")

    benchmark_hashes = {}
    for filename in sorted(BENCHMARK_FILES):
        path = directory / filename
        if not path.exists():
            continue
        benchmark = read(path)
        audit.equal(benchmark["source_report_sha256"], file_hash(directory / "report.json"), "benchmark report binding")
        audit.equal(benchmark["policy_sha256"], file_hash(directory / "policy.json"), "benchmark policy binding")
        audit_benchmark(audit, benchmark, report, policy, local["test"], splits["test"], artifacts["test-features.json"])
        benchmark_hashes[filename] = file_hash(path)

    public_hashes = {}
    analysis_software = {}
    if args.public is not None:
        files = {path.name for path in args.public.iterdir()}
        if args.output is not None and args.output.resolve().parent == args.public.resolve():
            files.discard(args.output.name)
        expected = PUBLIC_FILES | {"report.json"} | set(benchmark_hashes)
        if "analysis.json" in files:
            expected.add("analysis.json")
            analysis = read(args.public / "analysis.json")
            audit.equal(analysis["source_report_sha256"], file_hash(directory / "report.json"), "paired analysis report binding")
            spec = importlib.util.spec_from_file_location("_verifier_audit_analysis", ROOT / "scripts/analyze_document_verifier.py")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            replayed = module.analyze(report, file_hash(directory / "report.json"), seed=analysis["bootstrap"]["seed"], repetitions=analysis["bootstrap"]["attempted_replicates"])
            audit.equal({key: value for key, value in analysis.items() if key != "completed_utc"}, replayed, "paired analysis bootstrap replay")
            analysis_software = {name: digest((ROOT / "scripts" / name).read_text(encoding="utf-8")) for name in ("analyze_document_verifier.py", "analyze_document_v2.py")}
        audit.equal(files, expected, "public folder exact file allowlist")
        for filename in sorted(files):
            if filename != "analysis.json":
                audit.equal(file_hash(args.public / filename), file_hash(directory / filename), "public export identical " + filename)
            audit_public_privacy(audit, read(args.public / filename))
            public_hashes[filename] = file_hash(args.public / filename)
    else:
        for value in [report, *artifacts.values(), *[read(directory / filename) for filename in benchmark_hashes]]:
            audit_public_privacy(audit, value)
    audit.check("torch" not in sys.modules and "transformers" not in sys.modules, "audit imported no model runtime")
    return {"format": "cpu-document-verifier-audit-v1", "passed": True, "checks": audit.checks,
            "split_counts": {name: len(splits[name]) for name in SPLITS}, "positive_train_labels": sum(labels),
            "historical_exclusions": {"unique_ids": len(old_ids), "unique_contexts": len(old_contexts)},
            "report_sha256": file_hash(directory / "report.json"), "audit_script_sha256": file_hash(Path(__file__)),
            "prior_manifest_sha256": prior_hashes, "benchmark_sha256": benchmark_hashes, "public_sha256": public_hashes,
            "supplemental_analysis_software_sha256": analysis_software,
            "limitations": ["No language-model inference is repeated. Saved confidence summaries are checked for consistency but cannot be independently reconstructed without token logits.",
                            "Saved timestamps and frozen-source ordering support policy-before-test chronology; timestamps alone are not cryptographic proof of execution order.",
                            "Recorded token counts are checked against limits; this audit does not import a tokenizer or retokenize the corpus."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "artifacts/document-verifier-run")
    parser.add_argument("--prepared", type=Path, default=ROOT / "artifacts/document-verifier-data")
    parser.add_argument("--source", type=Path, default=ROOT / "artifacts/document-task-v1")
    parser.add_argument("--source-cache", type=Path, default=ROOT / ".cache/document-v2-data")
    parser.add_argument("--prior-manifests", type=Path, nargs="+", default=[ROOT / "results/document-split-manifest.json", ROOT / "results/document-v2-split-manifest.json"])
    parser.add_argument("--public", type=Path, help="Also check an exported public folder byte-for-byte.")
    parser.add_argument("--output", type=Path, help="Optional counts/checksums audit JSON; refuses to overwrite different bytes.")
    args = parser.parse_args()
    result = run(args)
    payload = (json.dumps(result, indent=2, allow_nan=False) + "\n").encode("utf-8")
    if args.output:
        if args.output.exists() and args.output.read_bytes() != payload:
            raise FileExistsError("Existing audit output is preserved; use a new output file.")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_bytes(payload)
    print(payload.decode("utf-8"), end="")


if __name__ == "__main__":
    main()
