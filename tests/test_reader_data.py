"""Fresh corpus, full passages, exact native spans and capped balanced sampling."""
import json
import re

import pytest

from cpu_ai_lab import reader_data as task


def config(**changes):
    return {"model_id": "fixture", "revision": "fixed", "reader_model_id": "reader",
            "reader_revision": "fixed", "seed": 20261101, "train_size": 4,
            "validation_size": 4, "test_size": 4, "short_context_tokens": 5,
            "max_context_tokens": 100, "max_prompt_tokens": 500, "max_answer_tokens": 8,
            "reader_max_length": 512, "reader_max_answer_tokens": 32,
            "max_questions_per_movie": 4, **changes}


class Generator:
    chat_template = None

    def encode(self, text, add_special_tokens=False):
        return list(range(len(text.split())))


class Encoded(dict):
    def __init__(self, offsets, sequences):
        super().__init__(input_ids=list(range(len(offsets))), offset_mapping=offsets)
        self.sequences = sequences

    def sequence_ids(self):
        return self.sequences


class Reader:
    def __call__(self, question, context, return_offsets_mapping, truncation):
        assert return_offsets_mapping is True and truncation is False
        first = [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", question)]
        second = [(m.start(), m.end()) for m in re.finditer(r"\w+|[^\w\s]", context)]
        return Encoded([(0, 0)] + first + [(0, 0)] + second + [(0, 0)],
                       [None] + [0] * len(first) + [None] + [1] * len(second) + [None])


def document(movie="m", context="Paris is a city.", questions=None):
    return {"id": movie, "title": "Fixture film", "plot": context,
            "qa": questions or [{"id": movie + "-yes", "question": "Which city?", "answers": ["Paris"], "no_answer": False},
                                {"id": movie + "-no", "question": "Who is mayor?", "answers": [], "no_answer": True}]}


def empty_controls():
    return {"all_prior_ids": [], "all_prior_context_sha256": []}


def eligible(documents, cfg=None, controls=None):
    return task.eligible_records(documents, Generator(), Reader(), cfg or config(), "fixture", controls or empty_controls())


def fixture_splits():
    result = {}
    for name in task.SPLITS:
        questions = [{"id": name + str(i), "question": f"Question {i}?", "answers": [] if i % 2 else ["Paris"], "no_answer": bool(i % 2)} for i in range(4)]
        result[name] = eligible([document(name, f"Paris is in the {name} plot.", questions)])[0]
    return result


def historical():
    return {name: [{"id": "old-" + name, "context_sha256": task.sha256(name)}] for name in task.SPLITS}


def controls():
    old = historical()
    return task.exclusion_controls([old], [task.sha256(json.dumps(old))])


def save_fixture(path):
    splits, cfg = fixture_splits(), config()
    (path / "data").mkdir()
    hashes = {}
    for name, rows in splits.items():
        target = path / "data" / (name + ".jsonl")
        target.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
        hashes[name] = task.sha256(target.read_bytes())
    task.write_json(path / "split_manifest.json", task.public_manifest(splits))
    control = controls()
    task.write_json(path / "data_provenance.json", {
        "data_config": cfg, "historical_exclusion_controls": control,
        "prior_manifest_sha256": [row["sha256"] for row in control["prior_manifests"]],
        "local_data_sha256": hashes,
        "split_manifest_sha256": task.sha256((path / "split_manifest.json").read_bytes()),
    })
    return splits, cfg


def test_full_context_native_missing_and_all_original_references_are_preserved():
    docs = [document()]
    docs[0]["qa"][0]["answers"] = ["the capital of France", "Paris", "Paris"]
    rows, stats = eligible(docs)
    positive, missing = rows
    assert positive["context"] == docs[0]["plot"]
    assert positive["references"] == ["the capital of France", "Paris", "Paris"]
    assert positive["response"] == "Paris"
    assert positive["literal_answer_spans"] == [{"text": "Paris", "start": 0, "end": 5}]
    assert positive["prompt"] == task.prompt_text(docs[0]["plot"], docs[0]["qa"][0]["question"])
    assert positive["article_sha256"] == task.sha256(docs[0]["id"])
    assert missing["response"] == task.MARKER and missing["references"] == missing["literal_answer_spans"] == []
    assert stats["eligible_questions"] == 2


def test_nonliteral_or_unaligned_positive_is_excluded_never_relabelled_missing():
    docs = [document(context="Parisians live here.")]
    rows, stats = eligible(docs)
    assert len(rows) == 1 and rows[0]["is_impossible"] is True
    assert stats["rejected_no_bounded_aligned_literal_span"] == 1


def test_all_aligned_occurrences_retained_in_deterministic_order():
    doc = document(context="Paris is Paris.")
    positive = eligible([doc])[0][0]
    assert positive["literal_answer_spans"] == [{"text": "Paris", "start": 0, "end": 5}, {"text": "Paris", "start": 9, "end": 14}]


@pytest.mark.parametrize("field,value,stat", [
    ("max_context_tokens", 2, "rejected_complete_context_length"),
    ("max_prompt_tokens", 2, "rejected_generator_prompt_length"),
    ("reader_max_length", 2, "rejected_reader_joint_length"),
])
def test_complete_inputs_exceeding_either_model_budget_are_excluded(field, value, stat):
    rows, stats = eligible([document()], config(**{field: value}))
    assert rows == [] and stats[stat] == 2


def test_span_bounds_apply_to_both_tokenizers():
    doc = document(context="New York is a city.")
    doc["qa"][0]["answers"] = ["New York"]
    for cfg in (config(max_answer_tokens=1), config(reader_max_answer_tokens=1)):
        rows, _ = eligible([doc], cfg)
        assert len(rows) == 1 and rows[0]["is_impossible"]


@pytest.mark.parametrize("mutation", ["integer_label", "negative_with_answer", "positive_without_answer"])
def test_inconsistent_native_labels_are_rejected(mutation):
    doc = document()
    if mutation == "integer_label":
        doc["qa"][0]["no_answer"] = 0
    elif mutation == "negative_with_answer":
        doc["qa"][1]["answers"] = ["Paris"]
    else:
        doc["qa"][0]["answers"] = []
    with pytest.raises(ValueError, match="native"):
        eligible([doc])


def test_duplicate_question_text_within_movie_is_removed():
    doc = document()
    duplicate = {**doc["qa"][0], "id": "another-native-id"}
    doc["qa"].append(duplicate)
    rows, stats = eligible([doc])
    assert len(rows) == 2
    assert stats["rejected_duplicate_question_text_within_movie"] == 1


def test_historical_exclusions_match_raw_ids_namespaced_ids_and_contexts():
    doc = document()
    for identifiers in (["m-yes"], ["duorc:ParaphraseRC:m-yes"]):
        rows, stats = eligible([doc], controls={"all_prior_ids": identifiers, "all_prior_context_sha256": []})
        assert len(rows) == 1 and rows[0]["is_impossible"]
        assert stats["rejected_historical_id_or_context"] == 1
    rows, _ = eligible([doc], controls={"all_prior_ids": [], "all_prior_context_sha256": [task.sha256(doc["plot"])]})
    assert rows == []


def test_max_flow_recovers_balanced_solution_from_flexible_movie_competition():
    # Two flexible movies must supply negatives, leaving positive-only movies for positives.
    rows = []
    for movie in ("flexible-a", "flexible-b", "positive-c", "positive-d"):
        doc = document(movie, "Paris is " + movie + ".")
        if movie.startswith("positive"):
            doc["qa"] = doc["qa"][:1]
        rows.extend(eligible([doc])[0])
    for seed in range(10):
        chosen = task.sample_balanced(rows, 4, 1, seed)
        assert len(chosen) == 4 and sum(row["is_impossible"] for row in chosen) == 2
        assert len({row["movie_id"] for row in chosen}) == 4
        assert task.sample_balanced(rows, 4, 1, seed) == chosen


def test_insufficient_movie_capacity_fails_instead_of_weakening_cap():
    rows = fixture_splits()["train"]
    with pytest.raises(ValueError, match="Insufficient"):
        task.sample_balanced(rows, 4, 2, 7)


def test_valid_balanced_splits_allow_repeated_context_within_movie_only():
    splits = fixture_splits()
    task.validate_splits(splits, config())
    assert len({row["context_sha256"] for row in splits["train"]}) == 1


@pytest.mark.parametrize("kind", ["movie", "context", "id"])
def test_cross_split_group_or_question_leakage_is_rejected(kind):
    splits = fixture_splits()
    key = {"movie": "movie_sha256", "context": "context_sha256", "id": "id"}[kind]
    splits["test"][0][key] = splits["train"][0][key]
    with pytest.raises(ValueError, match="leakage"):
        task.validate_splits(splits, config())


def test_invalid_supervision_and_private_public_fields_are_checked():
    splits = fixture_splits()
    manifest = task.public_manifest(splits)
    forbidden = {"movie_id", "article", "context", "question", "references", "response", "text", "prompt"}
    assert not (forbidden & set(manifest["train"][0]))
    assert set(manifest["train"][0]["literal_answer_spans"][0]) == {"text_sha256", "start", "end"}
    splits["train"][0]["literal_answer_spans"][0]["start"] = 1
    with pytest.raises(ValueError, match="unchanged original gold"):
        task.validate_splits(splits, config())


def test_prepared_reload_binds_both_model_revisions_and_preserves_extra_runtime_fields(tmp_path):
    splits, cfg = save_fixture(tmp_path)
    assert task.load_prepared(tmp_path, {**cfg, "learning_rate": 0.002})[0] == splits
    for key in ("revision", "reader_revision", "seed"):
        changed = {**cfg, key: 999 if key == "seed" else "different"}
        with pytest.raises(ValueError, match="configuration differs"):
            task.load_prepared(tmp_path, changed)


@pytest.mark.parametrize("tamper", ["data", "manifest", "prior_binding", "exclusion"])
def test_prepared_reload_checks_data_manifest_and_historical_exclusions(tmp_path, tamper):
    splits, cfg = save_fixture(tmp_path)
    if tamper == "data":
        (tmp_path / "data/test.jsonl").write_text("{}\n", encoding="utf-8")
    elif tamper == "manifest":
        (tmp_path / "split_manifest.json").write_text("{}", encoding="utf-8")
    else:
        path = tmp_path / "data_provenance.json"
        value = json.loads(path.read_bytes())
        if tamper == "prior_binding":
            value["prior_manifest_sha256"] = ["changed"]
        else:
            value["historical_exclusion_controls"]["all_prior_context_sha256"].append(splits["train"][0]["context_sha256"])
        task.write_json(path, value)
    with pytest.raises(ValueError, match="checksum|historical"):
        task.load_prepared(tmp_path, cfg)


def test_existing_output_is_preserved_before_model_or_network_access(tmp_path):
    (tmp_path / "existing.txt").write_text("preserved", encoding="utf-8")
    with pytest.raises(FileExistsError, match="preserved"):
        task.prepare(config(), tmp_path, tmp_path / "cache", False, [])


def test_source_checksum_is_checked_before_parsing(tmp_path):
    (tmp_path / "ParaphraseRC_train.json").write_text("[]", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum mismatch"):
        task.read_sources(tmp_path, False)


def test_all_historical_split_records_are_included():
    control = controls()
    assert len(control["all_prior_ids"]) == 3
    assert len(control["all_prior_context_sha256"]) == 3
    assert control["prior_record_count_with_repetition"] == 3
    with pytest.raises(ValueError, match="Distinct ordered"):
        task.exclusion_controls([historical(), historical()], ["same", "same"])
    with pytest.raises(ValueError, match="three nonempty"):
        task.exclusion_controls([{"train": []}], ["one"])


def test_config_balances_answerability_without_requiring_four_equal_strata():
    task.validate_config(config(train_size=6))
    with pytest.raises(ValueError, match="even size"):
        task.validate_config(config(train_size=5))
