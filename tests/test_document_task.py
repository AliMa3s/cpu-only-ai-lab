"""Document split, answer metric, batching and exact training-objective checks."""
import json

import pytest
import torch
from peft import LoraConfig, get_peft_model
from transformers import LlamaConfig, LlamaForCausalLM

from cpu_ai_lab import document_task as task
from cpu_ai_lab.training import collate_batch


def row(article, passage, identifier=None):
    identifier = identifier or f"{article}-{passage}"
    context = f"Passage {passage} for article {article}: Paris."
    return {
        "id": identifier, "article": article, "source": "fixture",
        "context": context, "question": "Which city?", "prompt": "prompt",
        "response": "Paris", "references": ["Paris"],
        "article_sha256": task.sha256(article.encode()),
        "context_sha256": task.sha256(context.encode()),
        "question_sha256": task.sha256(b"Which city?"),
        "prompt_tokens": 30, "context_tokens": 10, "reference_token_counts": [1],
    }


def test_squad_scoring_normalization_and_multiple_references():
    assert task.normalize_answer("  The, Eiffel Tower!  ") == "eiffel tower"
    assert task.score_answer("the Eiffel Tower!", ["Paris", "Eiffel Tower"]) == {
        "exact_match": 1.0, "token_f1": 1.0,
    }
    assert task.score_answer("New York", ["York City"])["token_f1"] == 0.5
    assert task.score_answer("", ["Paris"]) == {"exact_match": 0.0, "token_f1": 0.0}
    # Repeated prediction tokens count toward the precision denominator.
    assert task.score_answer("Paris Paris", ["Paris"])["token_f1"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        task.score_answer("anything", [])


def test_sampling_is_deterministic_balanced_and_one_question_per_passage():
    rows = [row(article, passage, f"{article}-{passage}-{question}")
            for article in ("A", "B", "C") for passage in range(3) for question in range(2)]
    selected = task.sample_articles(rows, 6, 42)
    assert selected == task.sample_articles(rows, 6, 42)
    assert len({item["context_sha256"] for item in selected}) == 6
    assert {article: sum(item["article"] == article for item in selected) for article in ("A", "B", "C")} == {
        "A": 2, "B": 2, "C": 2,
    }
    with pytest.raises(ValueError, match="distinct eligible passages"):
        task.sample_articles(rows, 10, 42)


def test_article_disjoint_splits_and_test_stability():
    train = [row(article, passage) for article in ("A", "B", "C", "D") for passage in range(3)]
    dev = [row("E", passage) for passage in range(3)]
    config = {"seed": 23, "validation_articles": 1, "train_size": 5, "validation_size": 2, "test_size": 2}
    splits, selection = task.select_splits(train, dev, config)
    task.validate_splits(splits)
    smaller, _ = task.select_splits(train, dev, {**config, "train_size": 3})
    assert smaller["test"] == splits["test"]
    assert smaller["validation"] == splits["validation"]
    assert selection["selected_article_counts"]["validation"] == 1
    for name in ("train", "validation"):
        assert all(item["article"] != "E" for item in splits[name])


@pytest.mark.parametrize("field", ["article", "id", "context_sha256"])
def test_cross_split_leakage_is_rejected(field):
    splits = {"train": [row("A", 0)], "validation": [row("B", 0)], "test": [row("C", 0)]}
    splits["test"][0][field] = splits["train"][0][field]
    with pytest.raises(ValueError, match="leakage"):
        task.validate_splits(splits)


class WordTokenizer:
    def encode(self, text, **kwargs):
        return list(range(len(text.split())))


def source_document(context="Paris is in France.", answers=None):
    return [{"title": "Geography", "paragraphs": [{"context": context, "qas": [{
        "id": "source-id", "question": "Which city is in France?",
        "answers": answers or [{"answer_start": 0, "text": "Paris"}],
    }]}]}]


def eligibility_config(**overrides):
    return {"max_context_tokens": 10, "max_prompt_tokens": 100,
            "max_answer_tokens": 2, "max_sequence_length": 110, **overrides}


def test_eligibility_preserves_entire_context_and_excludes_long_passages(monkeypatch):
    monkeypatch.setattr(task, "render_prompt", lambda tokenizer, example: example["prompt"])
    source = source_document()
    eligible, stats = task.eligible_records(source, WordTokenizer(), eligibility_config(), "fixture")
    assert eligible[0]["context"] == source[0]["paragraphs"][0]["context"]
    assert eligible[0]["context"] in eligible[0]["prompt"]
    assert stats["eligible_questions"] == 1
    excluded, stats = task.eligible_records(source, WordTokenizer(), eligibility_config(max_context_tokens=3), "fixture")
    assert not excluded
    assert stats["rejected_context_length"] == 1
    assert source[0]["paragraphs"][0]["context"] == "Paris is in France."


def test_all_answer_variants_must_fit_and_declared_spans_must_match(monkeypatch):
    monkeypatch.setattr(task, "render_prompt", lambda tokenizer, example: example["prompt"])
    source = source_document(answers=[{"answer_start": 0, "text": "Paris"}, {"answer_start": 0, "text": "Paris is in France"}])
    eligible, stats = task.eligible_records(source, WordTokenizer(), eligibility_config(), "fixture")
    assert not eligible
    assert stats["rejected_answer_length_or_empty_normalization"] == 1
    with pytest.raises(ValueError, match="declared source span"):
        task.eligible_records(source_document(answers=[{"answer_start": 2, "text": "Paris"}]), WordTokenizer(), eligibility_config(), "fixture")


def test_manifest_contains_identifiers_and_hashes_but_no_raw_text():
    manifest = task.public_manifest({"train": [row("Geography", 1)]})
    serialized = json.dumps(manifest)
    for forbidden in ("Which city?", "Passage 1 for article", "Paris"):
        assert forbidden not in serialized
    assert manifest["train"][0]["reference_sha256"] == [task.sha256(b"Paris")]


def test_left_padding_masks_and_per_row_eos_trimming():
    batch = task.left_pad([[4, 5, 6], [7]], 0)
    assert batch["input_ids"].tolist() == [[4, 5, 6], [0, 0, 7]]
    assert batch["attention_mask"].tolist() == [[1, 1, 1], [0, 0, 1]]
    assert task.trim_generated([8, 2, 0, 0], 2) == [8, 2]
    assert task.trim_generated([8, 9], 2) == [8, 9]


def test_generation_restores_original_order_and_counts_real_output_tokens(monkeypatch):
    monkeypatch.setattr(task, "render_prompt", lambda tokenizer, example: example["prompt"])

    class Tokenizer:
        pad_token_id, eos_token_id = 0, 2

        def encode(self, text, **kwargs):
            return [int(item) for item in text.split()]

        def decode(self, ids, **kwargs):
            return " ".join({11: "Paris", 12: "Rome", 13: "Berlin"}[i] for i in ids if i not in (0, 2))

    class Generator:
        def eval(self):
            return self

        def generate(self, input_ids, attention_mask, **kwargs):
            assert kwargs["do_sample"] is False
            assert kwargs["max_new_tokens"] == 3
            answers = {5: [11, 2, 0], 6: [12, 2, 0], 7: [13, 13, 13]}
            tokens = torch.tensor([answers[int(ids[-1])] for ids in input_ids])
            return torch.cat((input_ids, tokens), dim=1)

    rows = [row("A", 1), row("B", 2), row("C", 3)]
    for item, prompt, answer in zip(rows, ("9 9 5", "6", "9 7"), ("Paris", "Rome", "Berlin")):
        item.update(prompt=prompt, references=[answer], context=answer)
    metrics, predictions = task.generation_evaluation(Generator(), Tokenizer(), rows, 2, 3)
    assert [item["id"] for item in predictions] == [item["id"] for item in rows]
    assert [item["prediction"] for item in predictions] == ["Paris", "Rome", "Berlin Berlin Berlin"]
    assert metrics["generated_tokens_including_eos"] == 7
    assert metrics["ended_with_eos"] == 2
    assert metrics["exact_match"] == pytest.approx(2 / 3)
    assert metrics["token_f1"] == pytest.approx(5 / 6)


def test_response_only_projection_matches_full_loss_and_lora_gradients():
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(71)
        config = LlamaConfig(vocab_size=32, hidden_size=16, intermediate_size=32,
                             num_hidden_layers=1, num_attention_heads=2, num_key_value_heads=2,
                             max_position_embeddings=64, attention_dropout=0.0, pad_token_id=0)
        model = get_peft_model(LlamaForCausalLM(config), LoraConfig(
            task_type="CAUSAL_LM", r=2, lora_alpha=4, lora_dropout=0.0,
            target_modules=["q_proj", "v_proj"], bias="none"))
        # Nonzero B also exercises A gradients, not just initially-zero A gradients.
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_B" in name:
                    parameter.normal_(0, 0.02)
        model.eval()
        batch = collate_batch([
            {"input_ids": [1, 4, 5, 8, 9, 2], "labels": [-100, -100, -100, 8, 9, 2]},
            {"input_ids": [1, 6, 10, 2], "labels": [-100, -100, 10, 2]},
        ], 0)
        standard_loss = model(**batch, use_cache=False).loss
        standard_loss.backward()
        standard_gradients = {name: parameter.grad.detach().clone() for name, parameter in model.named_parameters() if parameter.requires_grad}
        model.zero_grad(set_to_none=True)
        optimized_loss, targets = task.response_loss(model, batch)
        optimized_loss.backward()
        assert targets == 5
        torch.testing.assert_close(optimized_loss, standard_loss, rtol=1e-6, atol=1e-7)
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter.grad, standard_gradients[name], rtol=2e-5, atol=1e-7)
        empty = {**batch, "labels": torch.full_like(batch["labels"], -100)}
        with pytest.raises(ValueError, match="shifted response targets"):
            task.response_loss(model, empty)
    finally:
        torch.set_num_threads(previous_threads)


def test_prepared_data_hashes_reject_modified_rows(tmp_path):
    config = {key: index for index, key in enumerate(task.DATA_KEYS)}
    splits = {"train": [row("A", 0)], "validation": [row("B", 0)], "test": [row("C", 0)]}
    data = tmp_path / "data"
    data.mkdir()
    hashes = {}
    for name, values in splits.items():
        path = data / f"{name}.jsonl"
        path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")
        hashes[name] = task.sha256(path.read_bytes())
    manifest = tmp_path / "split_manifest.json"
    manifest.write_text(json.dumps(task.public_manifest(splits)), encoding="utf-8")
    provenance = {"data_config": config, "split_manifest_sha256": task.sha256(manifest.read_bytes()), "local_data_sha256": hashes}
    (tmp_path / "data_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    loaded, _ = task.load_prepared(tmp_path, config)
    assert loaded == splits
    (data / "test.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="local data checksum"):
        task.load_prepared(tmp_path, config)


def saved_run_fixture(directory):
    directory.mkdir()
    for filename in ("resolved_config.json", "training_curve.json", "data_provenance.json", "split_manifest.json"):
        (directory / filename).write_bytes(b"{\r\n  \"fixture\": true\r\n}\r\n")
    for name in ("adapter", "tokenizer"):
        (directory / name).mkdir()
        (directory / name / "config.json").write_bytes(b"{}")
    (directory / "run_status.json").write_text(json.dumps({"status": "complete"}))
    metrics = {
        "resolved_config_sha256": task.sha256((directory / "resolved_config.json").read_bytes()),
        "training_curve_sha256": task.sha256((directory / "training_curve.json").read_bytes()),
        "data_provenance_sha256": task.sha256((directory / "data_provenance.json").read_bytes()),
        "dataset": {"split_manifest_sha256": task.sha256((directory / "split_manifest.json").read_bytes())},
        "adapter_sha256": task.directory_hashes(directory / "adapter"),
        "tokenizer_sha256": task.directory_hashes(directory / "tokenizer"),
    }
    (directory / "metrics.json").write_text(json.dumps(metrics))
    return metrics


@pytest.mark.parametrize("relative", ["resolved_config.json", "tokenizer/config.json", "adapter/config.json"])
def test_replay_verifies_config_tokenizer_and_adapter_bytes(tmp_path, relative):
    directory = tmp_path / "run"
    metrics = saved_run_fixture(directory)
    task.verify_saved_artifacts(directory, metrics)
    (directory / relative).write_bytes(b"changed")
    with pytest.raises(ValueError, match="checksum mismatch"):
        task.verify_saved_artifacts(directory, metrics)


def test_public_export_preserves_bytes_and_excludes_local_text(tmp_path):
    directory, destination = tmp_path / "run", tmp_path / "public"
    saved_run_fixture(directory)
    (directory / "predictions-after-local.json").write_text('"private answer"')
    (directory / "data").mkdir()
    (directory / "data" / "test.jsonl").write_text('"private passage"')
    report = task.export_public(directory, destination)
    assert report["raw_passages_questions_answers_or_predictions_exported"] is False
    assert {path.name for path in destination.iterdir()} == {
        "document-metrics.json", "document-config.json", "document-training-curve.json",
        "document-data-provenance.json", "document-split-manifest.json",
    }
    assert (destination / "document-config.json").read_bytes() == (directory / "resolved_config.json").read_bytes()
    assert all(b"private" not in path.read_bytes() for path in destination.iterdir())
    assert task.export_public(directory, destination) == report
    (destination / "document-config.json").write_bytes(b"previous published result")
    with pytest.raises(FileExistsError, match="different bytes"):
        task.export_public(directory, destination)
