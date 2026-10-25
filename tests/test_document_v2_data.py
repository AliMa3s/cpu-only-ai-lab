"""Native no-answer labels, complete contexts and inherited split controls."""
import json

import pytest

from cpu_ai_lab import document_v2_data as task


def config(**overrides):
    return {"model_id": "fixture", "revision": "fixed", "seed": 71,
            "train_size": 4, "validation_size": 4, "test_size": 4,
            "max_context_tokens": 8, "short_context_tokens": 4,
            "max_prompt_tokens": 100, "max_answer_tokens": 8,
            "max_sequence_length": 110, **overrides}


def row(article, group, index):
    impossible = group.startswith("missing_")
    length_group = group.split("_")[1]
    identifier = f"{article}-{group}-{index}"
    context = f"{identifier}: Paris."
    return {"id": identifier, "article": article, "source": "fixture",
            "context": context, "question": "Which city?", "prompt": "prompt",
            "response": task.MARKER if impossible else "Paris",
            "references": [] if impossible else ["Paris"],
            "is_impossible": impossible, "length_group": length_group,
            "article_sha256": task.sha256(article.encode()),
            "context_sha256": task.sha256(context.encode()),
            "question_sha256": task.sha256(b"Which city?"),
            "prompt_tokens": 20, "context_tokens": 3 if length_group == "short" else 6,
            "reference_token_counts": [] if impossible else [1]}


def rows(article, repeats=2):
    return [row(article, group, index) for group in task.STRATA for index in range(repeats)]


def controls():
    return {"prior_train_and_validation_article_sha256": [task.sha256(b"A"), task.sha256(b"OldValidation")],
            "prior_test_ids": ["previous-test"], "prior_test_context_sha256": [task.sha256(b"previous-context")],
            "prior_test_article_sha256": [task.sha256(b"C")]}


class WordTokenizer:
    def encode(self, text, **kwargs):
        return list(range(len(text.split())))


def document(context="Paris is a city.", impossible=False, answers=None, plausible=None):
    answer = [] if impossible else [{"text": "Paris", "answer_start": 0}]
    return [{"title": "Geography", "paragraphs": [{"context": context, "qas": [{
        "id": "source-id", "question": "Which city?", "is_impossible": impossible,
        "answers": answer if answers is None else answers,
        "plausible_answers": plausible or [],
    }]}]}]


@pytest.fixture(autouse=True)
def simple_render(monkeypatch):
    monkeypatch.setattr(task, "render_prompt", lambda tokenizer, example: example["prompt"])


def test_native_impossible_ignores_plausible_distractor_and_preserves_context():
    source = document(impossible=True, plausible=[{"text": "Paris", "answer_start": 0}])
    result, stats = task.eligible_records(source, WordTokenizer(), config(), "fixture")
    assert len(result) == 1
    item = result[0]
    assert item["response"] == "NOT_FOUND"
    assert item["references"] == []
    assert item["reference_token_counts"] == []
    assert item["context"] == source[0]["paragraphs"][0]["context"]
    assert item["context"] in item["prompt"]
    assert task.stratum(item) == "missing_short"
    assert stats["eligible_missing_short"] == 1
    longer, _ = task.eligible_records(document(context="Paris is a beautiful city.", impossible=True), WordTokenizer(), config(), "fixture")
    assert task.stratum(longer[0]) == "missing_long"


def test_answerable_uses_first_reference_and_checks_all_lengths():
    source = document(context="Paris and London are cities.", answers=[{"text": "Paris", "answer_start": 0}, {"text": "London", "answer_start": 10}])
    result, _ = task.eligible_records(source, WordTokenizer(), config(), "fixture")
    assert result[0]["references"] == ["Paris", "London"]
    assert result[0]["response"] == "Paris"
    assert not result[0]["is_impossible"]
    source[0]["paragraphs"][0]["qas"][0]["answers"][1] = {"text": "London are cities", "answer_start": 10}
    result, stats = task.eligible_records(source, WordTokenizer(), config(max_answer_tokens=2), "fixture")
    assert not result
    assert stats["rejected_answer_length_or_empty_normalization"] == 1


@pytest.mark.parametrize("source", [document(impossible=True, answers=[{"text": "Paris", "answer_start": 0}]), document(answers=[])])
def test_native_answerability_must_match_gold_answer_list(source):
    with pytest.raises(ValueError, match="answerability"):
        task.eligible_records(source, WordTokenizer(), config(), "fixture")


def test_source_span_and_marker_ambiguity_checks():
    with pytest.raises(ValueError, match="declared source span"):
        task.eligible_records(document(answers=[{"text": "Paris", "answer_start": 1}]), WordTokenizer(), config(), "fixture")
    result, stats = task.eligible_records(document(context="NOT_FOUND appears here.", answers=[{"text": "NOT_FOUND", "answer_start": 0}]), WordTokenizer(), config(), "fixture")
    assert result == []
    assert stats["rejected_ambiguous_marker_reference"] == 1


def test_no_cropping_even_when_answer_would_fit():
    source = document(context="Paris one two three four five six seven eight nine.")
    result, stats = task.eligible_records(source, WordTokenizer(), config(), "fixture")
    assert not result
    assert stats["rejected_context_length"] == 1
    source = document()
    result, stats = task.eligible_records(source, WordTokenizer(), config(max_prompt_tokens=5), "fixture")
    assert not result
    assert stats["rejected_prompt_length"] == 1


def test_sampling_balances_four_strata_and_shared_contexts():
    candidates = rows("A", 5) + rows("B", 5)
    # One paragraph has both an answerable and native impossible question.
    first = candidates[0]
    second = next(r for r in candidates if task.stratum(r) == "missing_short")
    second["context_sha256"] = first["context_sha256"]
    selected = task.sample_stratified(candidates, 16, 4)
    assert selected == task.sample_stratified(candidates, 16, 4)
    assert {key: sum(task.stratum(r) == key for r in selected) for key in task.STRATA} == dict.fromkeys(task.STRATA, 4)
    assert len({r["context_sha256"] for r in selected}) == 16
    with pytest.raises(ValueError, match="multiple of four"):
        task.sample_stratified(candidates, 15, 4)
    with pytest.raises(ValueError, match="Insufficient"):
        task.sample_stratified(rows("A", 1), 8, 4)


def test_split_separation_excludes_prior_articles_and_old_test_ids_or_contexts():
    current_controls = controls()
    train = rows("A", 3) + rows("B", 3) + rows("OldValidation", 3)
    dev = rows("C", 4)
    dev[0]["id"] = "previous-test"
    dev[1]["context_sha256"] = task.sha256(b"previous-context")
    splits, selection = task.select_splits(train, dev, config(), current_controls)
    assert all(r["article"] == "B" for r in splits["validation"])
    assert all(r["article"] in ("A", "OldValidation") for r in splits["train"])
    assert all(r["id"] != "previous-test" and r["context_sha256"] != task.sha256(b"previous-context") for r in splits["test"])
    assert selection["validation_article_pool_count"] == 1
    assert selection["prior_test_rows_excluded_by_id_or_context"] == 2
    assert selection["selected_test_articles_previously_evaluated"] == 1
    task.validate_splits(splits)
    more_train, _ = task.select_splits(train, dev, config(train_size=8), current_controls)
    assert more_train["validation"] == splits["validation"]
    assert more_train["test"] == splits["test"]


@pytest.mark.parametrize("field", ["article_sha256", "context_sha256", "id"])
def test_cross_split_leakage_rejected(field):
    splits = {"train": rows("A", 1), "validation": rows("B", 1), "test": rows("C", 1)}
    splits["test"][0][field] = splits["train"][0][field]
    with pytest.raises(ValueError, match="leakage"):
        task.validate_splits(splits)


def test_manifest_does_not_publish_raw_text():
    manifest = task.public_manifest({"test": rows("C", 1)})
    for item in manifest["test"]:
        assert not {"article", "context", "question", "prompt", "response", "references"} & item.keys()
        assert {"is_impossible", "length_group", "reference_sha256", "response_sha256"} <= item.keys()


def save_fixture(directory, current_config=None):
    current_config = current_config or config()
    splits = {"train": rows("A", 1), "validation": rows("B", 1), "test": rows("C", 1)}
    data = directory / "data"
    data.mkdir()
    hashes = {}
    for name, values in splits.items():
        path = data / f"{name}.jsonl"
        path.write_text("".join(json.dumps(value) + "\n" for value in values), encoding="utf-8")
        hashes[name] = task.sha256(path.read_bytes())
    manifest = directory / "split_manifest.json"
    manifest.write_text(json.dumps(task.public_manifest(splits)), encoding="utf-8")
    provenance = {"data_config": {key: current_config[key] for key in task.DATA_KEYS},
                  "local_data_sha256": hashes, "split_manifest_sha256": task.sha256(manifest.read_bytes()),
                  "prior_experiment_controls": controls()}
    (directory / "data_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    return splits


def test_prepared_reload_verifies_hashes_configuration_and_controls(tmp_path):
    splits = save_fixture(tmp_path)
    assert task.load_prepared(tmp_path, config())[0] == splits
    with pytest.raises(ValueError, match="configuration differs"):
        task.load_prepared(tmp_path, config(seed=72))
    path = tmp_path / "data/test.jsonl"
    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="local data checksum"):
        task.load_prepared(tmp_path, config())


def test_prepared_controls_reject_prior_article_reuse(tmp_path):
    save_fixture(tmp_path)
    path = tmp_path / "data_provenance.json"
    provenance = json.loads(path.read_text())
    provenance["prior_experiment_controls"]["prior_train_and_validation_article_sha256"].append(task.sha256(b"B"))
    path.write_text(json.dumps(provenance))
    with pytest.raises(ValueError, match="Held-out articles"):
        task.load_prepared(tmp_path, config())


def test_config_rejects_invalid_balancing_and_length_budgets():
    for overrides in ({"train_size": 5}, {"short_context_tokens": 8}, {"max_sequence_length": 105}, {"test_size": True}):
        with pytest.raises(ValueError):
            task.validate_config(config(**overrides))
