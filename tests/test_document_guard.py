"""Gold-free guard behavior, validation-only decisions and evidence integrity."""
import json
from pathlib import Path
import subprocess
import sys

import pytest

from cpu_ai_lab import document_guard as task


def test_guard_preserves_model_marker_and_normalized_span():
    model = task.guard_output("Paris is in France.", " NOT_FOUND \n")
    assert model["answer"] == "NOT_FOUND" and model["origin"] == "model"
    assert not model["guard_triggered"]
    answer = task.guard_output("The Eiffel Tower is in Paris.", "  Eiffel Tower!  ")
    assert answer["answer"] == "Eiffel Tower!" and not answer["abstained"]
    assert not answer["guard_triggered"]


@pytest.mark.parametrize("prediction", ["Atlantis", "", "...", "the"])
def test_guard_rejects_nonspans_and_normalized_empty_outputs(prediction):
    result = task.guard_output("Paris is in France.", prediction)
    assert result["answer"] == "NOT_FOUND" and result["origin"] == "guard"
    assert result["guard_triggered"]
    assert "does not prove" in result["reason"]


def test_literal_substring_is_not_semantic_correctness_or_token_matching():
    assert not task.guard_output("Paris is in France.", "ris")["abstained"]
    assert not task.guard_output("Paris and Rome are mentioned.", "Rome")["abstained"]
    # Disabled policy retains an unsupported answer, rather than quietly applying the candidate.
    result = task.policy_output("Paris is in France.", "Atlantis", False)
    assert result["answer"] == "Atlantis" and not result["guard_triggered"]


def fixture_row(split, index):
    impossible = index >= 2
    group = "short" if index % 2 == 0 else "long"
    context = f"Paris is in France. Context {split} {index}."
    question = "Which city is mentioned?" if not impossible else "What time is the meeting?"
    article = f"article-{split}-{index}"
    references = [] if impossible else ["Paris"]
    return {"id": f"{split}-{index}", "source": "fixture", "article": article,
            "context": context, "question": question,
            "prompt": task.PROMPT_INSTRUCTION + context + "\nQuestion: " + question,
            "response": "NOT_FOUND" if impossible else "Paris", "references": references,
            "is_impossible": impossible, "length_group": group,
            "article_sha256": task.digest(article.encode()), "context_sha256": task.digest(context.encode()),
            "question_sha256": task.digest(question.encode()), "prompt_tokens": 250 if group == "long" else 60,
            "context_tokens": 200 if group == "long" else 10, "reference_token_counts": [] if impossible else [1]}


def fixture_run(tmp_path, *, validation=None, test=None):
    run, prepared = tmp_path / "run", tmp_path / "data"
    run.mkdir(); (prepared / "data").mkdir(parents=True)
    config = {"model_id": "fixture/model", "revision": "r", "seed": 1,
              "train_size": 4, "validation_size": 4, "test_size": 4,
              "max_context_tokens": 512, "short_context_tokens": 192, "max_prompt_tokens": 608,
              "max_answer_tokens": 8, "max_sequence_length": 624, "generation_tokens": 16}
    rows = {name: [fixture_row(name, i) for i in range(4)] for name in ("validation", "test")}
    manifest = {name: task.public_manifest(values) for name, values in rows.items()}
    task.write(prepared / "split_manifest.json", manifest)
    task.write(run / "split_manifest.json", manifest)
    local_hashes = {}
    for name, values in rows.items():
        path = prepared / "data" / f"{name}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in values), encoding="utf-8")
        local_hashes[name] = task.digest(path.read_bytes())
    source = {"metrics_sha256": "a" * 64, "split_manifest_sha256": "b" * 64,
              "adapter_sha256": {"adapter_model.safetensors": "c" * 64}, "selected_step": 75,
              "all_saved_artifact_checksums_verified": True}
    provenance = {"data_config": {key: config[key] for key in task.DATA_KEYS},
                  "split_manifest_sha256": task.digest((prepared / "split_manifest.json").read_bytes()),
                  "prior_manifest_sha256": source["split_manifest_sha256"], "local_data_sha256": local_hashes}
    task.write(prepared / "data_provenance.json", provenance)
    task.write(run / "data_provenance.json", provenance)
    task.write(run / "resolved_config.json", config)
    metrics = {"format": "cpu-ai-lab-document-v2-v1", "source": source, "dataset": provenance,
               "resolved_config_sha256": task.digest((run / "resolved_config.json").read_bytes()),
               "data_provenance_sha256": task.digest((run / "data_provenance.json").read_bytes()),
               "local_prediction_sha256": {}, "before": {}}
    for name, predictions in (("validation", validation), ("test", test)):
        predictions = predictions or [" Paris ", "Paris", "Atlantis", "Atlantis"]
        records, local = [], []
        for row, prediction in zip(rows[name], predictions):
            record = {**task.scoring.score_prediction(row, prediction), "ended_with_eos": True,
                      "generated_tokens_including_eos": 3}
            records.append(record)
            local.append({**record, "question": row["question"], "references": row["references"], "prediction": prediction})
        filename = "validation-initial-local.json" if name == "validation" else "test-before-local.json"
        task.write(run / filename, local)
        metrics["local_prediction_sha256"][filename] = task.digest((run / filename).read_bytes())
        metrics["before"][name] = {"generation": {**task.scoring.summarize(records), "records": records}}
    task.write(run / "metrics.json", metrics)
    task.write(run / "run_status.json", {"status": "complete"})
    return run, prepared


def test_offline_guard_passes_only_validation_and_keeps_test_descriptive(tmp_path):
    run, prepared = fixture_run(tmp_path, test=["Moon", "Moon", "Paris", "Paris"])
    output = tmp_path / "guard"
    report = task.evaluate_run(run, prepared, output)
    assert report["validation_gate"]["accepted"]
    assert report["selected_policy"] == "warmstart_plus_span_guard"
    assert report["validation"]["candidate"]["exact_match"] == 1
    assert report["test"]["candidate"]["exact_match"] == 0
    assert report["test"]["selected"] == report["test"]["candidate"]
    assert not report["model_inference_performed"]
    assert report["validation"]["candidate"]["guard_triggers"]["by_answerability"]["missing"]["triggered"] == 2
    policy, loaded = task.load_policy(output / "policy.json")
    assert policy["guard_enabled"] and loaded == report
    for metric in report["test"].values():
        assert all(not {"context", "question", "references", "prediction", "answer", "response"} & row.keys()
                   for row in metric["records"])
    with pytest.raises(FileExistsError):
        task.evaluate_run(run, prepared, output)


def test_failed_validation_retains_original_policy_even_when_test_would_improve(tmp_path):
    run, prepared = fixture_run(tmp_path, validation=["Moon", "Moon", "Paris", "Paris"])
    report = task.evaluate_run(run, prepared, tmp_path / "guard")
    assert not report["validation_gate"]["accepted"]
    assert report["selected_policy"] == "unchanged_warmstart"
    assert report["test"]["candidate"]["exact_match"] > report["test"]["warmstart"]["exact_match"]
    assert report["test"]["selected"] == report["test"]["warmstart"]


@pytest.mark.parametrize("filename", ["validation-initial-local.json", "resolved_config.json", "split_manifest.json", "data_provenance.json"])
def test_source_evidence_tampering_is_rejected(tmp_path, filename):
    run, prepared = fixture_run(tmp_path)
    with (run / filename).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(ValueError, match="checksum"):
        task.evaluate_run(run, prepared, tmp_path / "guard")


def test_prediction_identity_is_rechecked_even_when_its_file_hash_is_updated(tmp_path):
    run, prepared = fixture_run(tmp_path)
    path = run / "validation-initial-local.json"
    local = task.read(path)
    local.reverse()
    task.write(path, local)
    metrics = task.read(run / "metrics.json")
    metrics["local_prediction_sha256"][path.name] = task.digest(path.read_bytes())
    task.write(run / "metrics.json", metrics)
    with pytest.raises(ValueError, match="identities/order"):
        task.evaluate_run(run, prepared, tmp_path / "guard")


def test_policy_file_hash_and_config_fingerprint_are_verified(tmp_path):
    run, prepared = fixture_run(tmp_path)
    output = tmp_path / "guard"
    task.evaluate_run(run, prepared, output)
    path = output / "policy.json"
    policy = task.read(path)
    policy["generation_config"]["generation_tokens"] = 100
    task.write(path, policy)
    with pytest.raises(ValueError, match="checksum"):
        task.load_policy(path)
    report = task.read(output / "report.json")
    report["policy_sha256"] = task.digest(path.read_bytes())
    task.write(output / "report.json", report)
    with pytest.raises(ValueError, match="configuration fingerprint"):
        task.load_policy(path)


def test_primary_must_be_complete_before_followup(tmp_path):
    run, prepared = fixture_run(tmp_path)
    task.write(run / "run_status.json", {"status": "running"})
    with pytest.raises(ValueError, match="must finish"):
        task.evaluate_run(run, prepared, tmp_path / "guard")


def test_live_output_distinguishes_guard_from_model_without_loading_a_model():
    class ModelAPI:
        @staticmethod
        def answer(model, tokenizer, config, context, question):
            return {"answer": "Atlantis", "prediction_sha256": task.digest(b"Atlantis")}
    result = task.live_answer(ModelAPI, None, None, {"generation_config": {}, "guard_enabled": True},
                              "Paris is in France.", "Which city?")
    assert result["abstention_origin"] == "guard" and result["guard_triggered"]
    assert result["answer"] == "NOT_FOUND"


def test_offline_module_imports_no_model_stack():
    result = subprocess.run([sys.executable, "-c", "import sys; import cpu_ai_lab.document_guard; assert 'torch' not in sys.modules; assert 'transformers' not in sys.modules"],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
