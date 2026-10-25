"""Validation-selected span guard over saved warmstart document-v2 generations.

Offline evaluation uses only the standard library. The rule never reads a gold
answer. Live generation imports the CPU model stack only when requested.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from . import document_v2_metrics as scoring


MARKER = "NOT_FOUND"
PROMPT_INSTRUCTION = ("Answer the question using the passage. Copy only the shortest answer span. "
                      "If the passage does not contain the answer, output exactly NOT_FOUND. "
                      "Do not add explanations.\nPassage:\n")
RULE = {
    "version": 1,
    "name": "normalized_complete_context_substring",
    "marker": MARKER,
    "normalization": "lowercase; remove ASCII punctuation; remove English articles a/an/the; collapse whitespace",
    "matching": "nonempty normalized answer is a literal substring of normalized complete context",
    "existing_exact_marker": "preserve as model abstention after stripping surrounding whitespace",
    "otherwise": "return NOT_FOUND with origin guard; no gold answers or labels are inputs",
}
DATA_KEYS = ("model_id", "revision", "seed", "train_size", "validation_size", "test_size",
             "max_context_tokens", "short_context_tokens", "max_prompt_tokens", "max_answer_tokens",
             "max_sequence_length")


def digest(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def fingerprint(value) -> str:
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode())


def read(path: Path):
    return json.loads(path.read_bytes())


def write(path: Path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def checked(path: Path, expected: str) -> bytes:
    payload = path.read_bytes()
    if digest(payload) != expected:
        raise ValueError(f"Artifact checksum mismatch: {path.name}")
    return payload


def guard_output(context: str, prediction: str) -> dict:
    """The frozen rule accepts only context and model output, never references."""
    if not isinstance(context, str) or not isinstance(prediction, str):
        raise TypeError("Context and prediction must be text.")
    stripped = prediction.strip()
    if stripped == MARKER:
        return {"answer": MARKER, "abstained": True, "origin": "model", "guard_triggered": False,
                "reason": "model_emitted_NOT_FOUND"}
    normalized = scoring.normalize_answer(stripped)
    if normalized and normalized in scoring.normalize_answer(context):
        return {"answer": stripped, "abstained": False, "origin": "model", "guard_triggered": False,
                "reason": "normalized_output_is_a_passage_substring"}
    return {"answer": MARKER, "abstained": True, "origin": "guard", "guard_triggered": True,
            "reason": "no_supported_span_extracted; this does not prove that the passage lacks an answer"}


def policy_output(context: str, prediction: str, enabled: bool) -> dict:
    if type(enabled) is not bool:
        raise ValueError("Policy enabled must be boolean.")
    if enabled:
        return guard_output(context, prediction)
    stripped = prediction.strip()
    return {"answer": stripped, "abstained": stripped == MARKER, "origin": "model",
            "guard_triggered": False, "reason": "unchanged_warmstart_policy; guard_not_selected"}


def public_manifest(rows):
    return [{"id": row["id"], "source": row["source"], "article_sha256": row["article_sha256"],
             "context_sha256": row["context_sha256"], "question_sha256": row["question_sha256"],
             "reference_sha256": [digest(text.encode()) for text in row["references"]],
             "response_sha256": digest(row["response"].encode()),
             "is_impossible": row["is_impossible"], "length_group": row["length_group"],
             "prompt_tokens": row["prompt_tokens"], "context_tokens": row["context_tokens"],
             "reference_token_counts": row["reference_token_counts"]} for row in rows]


def prepared_metadata(prepared: Path, config: dict, source: dict):
    provenance = json.loads(checked(prepared / "data_provenance.json", source["data_provenance_sha256"]))
    manifest = json.loads(checked(prepared / "split_manifest.json", source["split_manifest_sha256"]))
    if provenance["data_config"] != {key: config[key] for key in DATA_KEYS}:
        raise ValueError("Prepared configuration differs from the recorded generation configuration.")
    if provenance["split_manifest_sha256"] != source["split_manifest_sha256"]:
        raise ValueError("Prepared manifest binding differs.")
    if provenance["prior_manifest_sha256"] != source["warmstart"]["split_manifest_sha256"]:
        raise ValueError("Prepared exclusions differ from the warmstart source.")
    return provenance, manifest


def prepared_rows(prepared: Path, split: str, config: dict, provenance: dict, manifest: dict):
    if split not in ("validation", "test"):
        raise ValueError("Guard evaluation uses validation and test only.")
    payload = checked(prepared / "data" / f"{split}.jsonl", provenance["local_data_sha256"][split])
    rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
    if len(rows) != config[f"{split}_size"] or public_manifest(rows) != manifest[split]:
        raise ValueError("Prepared records differ from the public split manifest.")
    ids, contexts = set(), set()
    counts = Counter()
    for row in rows:
        if row["id"] in ids or row["context_sha256"] in contexts:
            raise ValueError("Duplicate held-out ID or context.")
        ids.add(row["id"]); contexts.add(row["context_sha256"])
        for key, value in (("context_sha256", row["context"]), ("question_sha256", row["question"]),
                           ("article_sha256", row["article"])):
            if digest(value.encode()) != row[key]:
                raise ValueError("Prepared source text does not match its identity hash.")
        if row["prompt"] != PROMPT_INSTRUCTION + row["context"] + "\nQuestion: " + row["question"]:
            raise ValueError("Prepared prompt differs from the frozen generation prompt.")
        group = ("missing" if row["is_impossible"] else "answerable") + "_" + row["length_group"]
        counts[group] += 1
    if set(counts) != set(scoring.STRATA) or len(set(counts.values())) != 1:
        raise ValueError("Guard evaluation requires the original four balanced strata.")
    return rows


def load_predictions(run: Path, split: str, rows: list, metrics: dict):
    name = "validation-initial-local.json" if split == "validation" else "test-before-local.json"
    predictions = json.loads(checked(run / name, metrics["local_prediction_sha256"][name]))
    expected = metrics["before"][split]["generation"]["records"]
    if len(predictions) != len(rows) or len(expected) != len(rows):
        raise ValueError("Warmstart prediction count differs from the split.")
    for row, local, record in zip(rows, predictions, expected, strict=True):
        if row["id"] != local["id"] or row["id"] != record["id"]:
            raise ValueError("Warmstart prediction identities/order differ.")
        if local["question"] != row["question"] or local["references"] != row["references"]:
            raise ValueError("Saved prediction source differs from the prepared row.")
        if any(local.get(key) != value for key, value in record.items()):
            raise ValueError("Saved local/public generation records differ.")
        recalculated = scoring.score_prediction(row, local["prediction"])
        if any(record.get(key) != value for key, value in recalculated.items()):
            raise ValueError("Warmstart text/hash/scores do not match recorded metrics.")
    return [item["prediction"] for item in predictions]


def evaluate_outputs(rows: list, predictions: list[str], enabled: bool):
    if len(rows) != len(predictions):
        raise ValueError("One prediction per evaluation row is required.")
    records = []
    for row, prediction in zip(rows, predictions, strict=True):
        output = policy_output(row["context"], prediction, enabled)
        records.append({**scoring.score_prediction(row, output["answer"]),
                        "model_prediction_sha256": digest(prediction.encode()),
                        "abstention_origin": output["origin"] if output["abstained"] else None,
                        "guard_triggered": output["guard_triggered"]})
    def triggers(subset):
        total = len(subset)
        return {"examples": total, "triggered": sum(r["guard_triggered"] for r in subset),
                "trigger_rate": sum(r["guard_triggered"] for r in subset) / total if total else None}
    groups = {name: [r for r in records if ("missing" if r["is_impossible"] else "answerable") + "_" + r["length_group"] == name]
              for name in scoring.STRATA}
    return {**scoring.summarize(records), "guard_triggers": {
        "overall": triggers(records),
        "by_answerability": {name: triggers([r for r in records if r["is_impossible"] == impossible])
                             for name, impossible in (("answerable", False), ("missing", True))},
        "by_length": {name: triggers([r for r in records if r["length_group"] == name]) for name in ("short", "long")},
        "by_stratum": {name: triggers(group) for name, group in groups.items()}}, "records": records}


def evaluate_run(run: Path, prepared: Path, output: Path):
    if read(run / "run_status.json").get("status") != "complete":
        raise ValueError("The primary experiment must finish before evaluating a follow-up policy.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError("Existing policy evidence is preserved; choose a new output directory.")
    started = time.perf_counter()
    metric_payload = (run / "metrics.json").read_bytes()
    metrics = json.loads(metric_payload)
    if metrics.get("format") != "cpu-ai-lab-document-v2-v1":
        raise ValueError("Expected a document-v2 experiment.")
    config = json.loads(checked(run / "resolved_config.json", metrics["resolved_config_sha256"]))
    checked(run / "data_provenance.json", metrics["data_provenance_sha256"])
    checked(run / "split_manifest.json", metrics["dataset"]["split_manifest_sha256"])
    source = {"document_v2_metrics_sha256": digest(metric_payload),
              "document_v2_config_sha256": metrics["resolved_config_sha256"],
              "data_provenance_sha256": metrics["data_provenance_sha256"],
              "split_manifest_sha256": metrics["dataset"]["split_manifest_sha256"],
              "warmstart": metrics["source"],
              "initial_prediction_sha256": {name: metrics["local_prediction_sha256"][name]
                  for name in ("validation-initial-local.json", "test-before-local.json")}}
    provenance, manifest = prepared_metadata(prepared, config, source)
    if provenance != metrics["dataset"]:
        raise ValueError("Prepared provenance differs from primary report.")
    # The policy is decided and saved before loading the local test predictions.
    rows = prepared_rows(prepared, "validation", config, provenance, manifest)
    predictions = load_predictions(run, "validation", rows, metrics)
    baseline = evaluate_outputs(rows, predictions, False)
    candidate = evaluate_outputs(rows, predictions, True)
    gate = scoring.checkpoint_gate(baseline, candidate)
    enabled = gate["accepted"]
    validation = {"warmstart": baseline, "candidate": candidate,
                  "selected": candidate if enabled else baseline}
    policy = {"format": "cpu-ai-lab-document-span-policy-v1", "rule": RULE,
              "rule_sha256": fingerprint(RULE), "prompt_sha256": digest(PROMPT_INSTRUCTION.encode()),
              "generation_config": config, "config_fingerprint_sha256": fingerprint(config),
              "source": source, "guard_enabled": enabled,
              "selected_policy": "warmstart_plus_span_guard" if enabled else "unchanged_warmstart",
              "validation_gate": gate, "selection_uses_test": False,
              "development": "Post-hoc fixed-rule follow-up proposed after the primary LoRA validation failures; no rule/threshold search or test selection."}
    output.mkdir(parents=True, exist_ok=True)
    write(output / "policy.json", policy)
    rows = prepared_rows(prepared, "test", config, provenance, manifest)
    predictions = load_predictions(run, "test", rows, metrics)
    before_test = evaluate_outputs(rows, predictions, False)
    candidate_test = evaluate_outputs(rows, predictions, True)
    report = {"format": "cpu-ai-lab-document-span-guard-v1", "status": "complete",
              "completed_utc": datetime.now(timezone.utc).isoformat(),
              "policy_sha256": digest((output / "policy.json").read_bytes()), "source": source,
              "selected_policy": policy["selected_policy"], "validation_gate": gate,
              "validation": validation, "test": {"warmstart": before_test, "candidate": candidate_test,
                  "selected": candidate_test if enabled else before_test},
              "model_inference_performed": False, "additional_model_training_performed": False,
              "offline_evaluation_seconds": time.perf_counter() - started,
              "limitations": ["This is an output-validity policy over the unchanged warmstart, not learned semantic abstention.",
                  "A guard-induced NOT_FOUND means no supported span was extracted; it does not prove no answer exists.",
                  "Normalized substring checks can accept wrong distractor spans and word fragments, and reject partially correct non-span outputs.",
                  "The fixed rule was proposed after observing primary-run validation failure. The same validation gate is reused; this is a post-hoc follow-up.",
                  "Test generations are rescored only after the validation decision is frozen; no test tuning or repeated policy search.",
                  "The filtered balanced historical dataset and reused article topics retain the primary experiment's limitations.",
                  "This offline timing excludes model generation and cannot establish an inference speedup."]}
    write(output / "report.json", report)
    return report


def load_policy(path: Path):
    policy, report = read(path), read(path.parent / "report.json")
    if policy.get("format") != "cpu-ai-lab-document-span-policy-v1" or report.get("status") != "complete":
        raise ValueError("A completed span-policy report is required.")
    if digest(path.read_bytes()) != report["policy_sha256"]:
        raise ValueError("Policy/report checksum binding differs.")
    if policy["rule"] != RULE or policy["rule_sha256"] != fingerprint(RULE):
        raise ValueError("Unsupported or modified guard rule.")
    if policy["prompt_sha256"] != digest(PROMPT_INSTRUCTION.encode()):
        raise ValueError("Generation prompt fingerprint differs.")
    if policy["config_fingerprint_sha256"] != fingerprint(policy["generation_config"]):
        raise ValueError("Generation configuration fingerprint differs.")
    if policy["source"] != report["source"] or policy["selected_policy"] != report["selected_policy"]:
        raise ValueError("Policy/report source or selection binding differs.")
    gate = scoring.checkpoint_gate(report["validation"]["warmstart"], report["validation"]["candidate"])
    if gate != report["validation_gate"] or gate != policy["validation_gate"] or type(policy["guard_enabled"]) is not bool or policy["guard_enabled"] != gate["accepted"]:
        raise ValueError("Saved policy disagrees with the validation gate.")
    expected_name = "warmstart_plus_span_guard" if gate["accepted"] else "unchanged_warmstart"
    if policy["selected_policy"] != expected_name or policy["selection_uses_test"] is not False:
        raise ValueError("Invalid policy selection contract.")
    return policy, report


def live_model(policy: dict, source_run: Path, threads: int):
    if type(threads) is not int or threads < 1:
        raise ValueError("CPU threads must be a positive integer.")
    import torch
    from . import document_v2 as model_api
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("A CPU-only PyTorch build is required.")
    if model_api.data.INSTRUCTION != PROMPT_INSTRUCTION:
        raise ValueError("Current model prompt differs from the policy prompt.")
    torch.set_num_threads(threads)
    model, tokenizer, _, source = model_api.load_source(policy["generation_config"], source_run)
    if source != policy["source"]["warmstart"]:
        raise ValueError("Live warmstart source differs from the policy's verified source.")
    model.eval()
    return model_api, model, tokenizer


def live_answer(model_api, model, tokenizer, policy, context, question):
    generated = model_api.answer(model, tokenizer, policy["generation_config"], context, question)
    output = policy_output(context, generated["answer"], policy["guard_enabled"])
    return {"answer": output["answer"], "abstained": output["abstained"],
            "abstention_origin": output["origin"] if output["abstained"] else None,
            "guard_enabled": policy["guard_enabled"], "guard_triggered": output["guard_triggered"],
            "reason": output["reason"], "device": "cpu",
            "model_prediction_sha256": generated["prediction_sha256"],
            "prediction_sha256": digest(output["answer"].encode())}


def replay(path: Path, source_run: Path, prepared: Path, threads: int):
    policy, report = load_policy(path)
    provenance, manifest = prepared_metadata(prepared, policy["generation_config"], policy["source"])
    rows = prepared_rows(prepared, "test", policy["generation_config"], provenance, manifest)
    selected, groups = [], set()
    for row in rows:
        group = ("missing" if row["is_impossible"] else "answerable") + "_" + row["length_group"]
        if group not in groups:
            selected.append((row, group)); groups.add(group)
    model_api, model, tokenizer = live_model(policy, source_run, threads)
    expected = {row["id"]: row for row in report["test"]["selected"]["records"]}
    records = []
    for row, group in selected:
        result = live_answer(model_api, model, tokenizer, policy, row["context"], row["question"])
        records.append({"id": row["id"], "stratum": group,
                        "prediction_sha256": result["prediction_sha256"],
                        "model_prediction_sha256": result["model_prediction_sha256"],
                        "matches_saved_model_generation": result["model_prediction_sha256"] == expected[row["id"]]["model_prediction_sha256"],
                        "matches_saved_selected_policy": result["prediction_sha256"] == expected[row["id"]]["prediction_sha256"],
                        "abstention_origin": result["abstention_origin"]})
    evidence = {"policy_sha256": digest(path.read_bytes()), "examples": len(records),
                "model_matches": sum(row["matches_saved_model_generation"] for row in records),
                "policy_matches": sum(row["matches_saved_selected_policy"] for row in records),
                "selection": "First test record from each stratum in immutable source order; replay only, no model or policy selection.",
                "records": records}
    write(path.parent / "reload_verification.json", evidence)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path)
    parser.add_argument("--prepared-data", type=Path, default=Path("artifacts/document-v2-data"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--load-policy", type=Path)
    parser.add_argument("--verify-reload", type=Path)
    parser.add_argument("--source-run", type=Path, default=Path("artifacts/document-task-v1"))
    parser.add_argument("--context")
    parser.add_argument("--question")
    parser.add_argument("--threads", type=int, default=12)
    args = parser.parse_args()
    if args.load_policy:
        if not args.context or not args.question:
            parser.error("--load-policy requires --context and --question")
        policy, _ = load_policy(args.load_policy)
        api, model, tokenizer = live_model(policy, args.source_run, args.threads)
        print(json.dumps(live_answer(api, model, tokenizer, policy, args.context, args.question)))
    elif args.verify_reload:
        print(json.dumps(replay(args.verify_reload, args.source_run, args.prepared_data, args.threads)))
    else:
        if args.run is None or args.output is None:
            parser.error("Offline evaluation requires --run and --output")
        report = evaluate_run(args.run, args.prepared_data, args.output)
        print(json.dumps({"selected_policy": report["selected_policy"], "validation_gate": report["validation_gate"],
                          "test": {name: {key: value for key, value in metric.items() if key != "records"}
                                   for name, metric in report["test"].items()}}, indent=2))


if __name__ == "__main__":
    main()
