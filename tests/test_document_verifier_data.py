"""Fresh verifier examples exclude every historical split, including training."""
import json

import pytest

from cpu_ai_lab import document_verifier_data as task


def config(**overrides):
    return {"model_id": "fixture", "revision": "fixed", "seed": 20261030,
            "train_size": 8, "validation_size": 8, "test_size": 8,
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
            "response": task.MARKER if impossible else "Paris", "references": [] if impossible else ["Paris"],
            "is_impossible": impossible, "length_group": length_group,
            "article_sha256": task.sha256(article.encode()), "context_sha256": task.sha256(context.encode()),
            "question_sha256": task.sha256(b"Which city?"), "prompt_tokens": 20,
            "context_tokens": 3 if length_group == "short" else 6,
            "reference_token_counts": [] if impossible else [1]}


def rows(article, count=5):
    return [row(article, group, index) for group in task.STRATA for index in range(count)]


def historical():
    first = {"train": [row("A", "answerable_short", 100)],
             "validation": [row("OldValidation", "answerable_long", 100)],
             "test": [row("C", "missing_short", 100)]}
    second = {"train": [row("A", "missing_long", 101)],
              "validation": [row("B", "missing_long", 101)],
              "test": [row("C", "answerable_short", 101)]}
    return [first, second]


def controls():
    manifests = historical()
    return task.exclusion_controls(manifests, [task.sha256(json.dumps(m).encode()) for m in manifests])


def test_all_historical_splits_contribute_id_and_context_exclusions():
    previous = historical()
    control = controls()
    candidates = rows("A")
    known = [r for manifest in previous for split in ("train", "validation", "test") for r in manifest[split]]
    for index, old in enumerate(known):
        candidates[index]["id"] = old["id"]
        candidates[index + len(known)]["context_sha256"] = old["context_sha256"]
    fresh, counts = task.fresh_rows(candidates, control)
    assert len(fresh) == len(candidates) - 2 * len(known)
    assert counts["questions_matching_prior_ids"] == len(known)
    assert counts["questions_matching_prior_contexts"] == len(known)
    assert counts["questions_excluded_by_either_rule"] == 2 * len(known)
    assert len(control["all_prior_ids"]) == len(known)


def test_first_manifest_defines_article_holdout_but_all_manifests_exclude_examples():
    control = controls()
    train = rows("A") + rows("B") + rows("OldValidation")
    train.extend(historical()[1]["validation"])
    dev = rows("C") + historical()[1]["test"]
    splits, selection = task.select_splits(train, dev, config(), control)
    assert all(r["article"] == "B" for r in splits["validation"])
    assert all(r["article"] in ("A", "OldValidation") for r in splits["train"])
    assert all(r["article"] == "C" for r in splits["test"])
    task.check_historical_exclusions(splits, control)
    task.validate_splits(splits)
    assert selection["validation_article_pool_count"] == 1
    assert selection["historical_article_reuse"]["validation"]["selected_articles_seen_in_any_prior_manifest"] == 1
    assert selection["historical_exclusion_counts"]["official_train"]["questions_excluded_by_either_rule"] == 1
    assert selection["historical_exclusion_counts"]["official_dev"]["questions_excluded_by_either_rule"] == 1
    larger, _ = task.select_splits(train, dev, config(train_size=12), control)
    assert larger["validation"] == splits["validation"]
    assert larger["test"] == splits["test"]


def test_insufficient_fresh_passages_fail_instead_of_weakening_exclusions():
    control = controls()
    train = rows("A", 1) + rows("B", 1)
    with pytest.raises(ValueError, match="Insufficient"):
        task.select_splits(train, rows("C"), config(), control)


@pytest.mark.parametrize("split", ["train", "validation", "test"])
@pytest.mark.parametrize("field", ["id", "context_sha256"])
def test_historical_overlap_rejected_in_every_new_split(split, field):
    splits = {"train": rows("A", 1), "validation": rows("B", 1), "test": rows("C", 1)}
    splits[split][0][field] = historical()[1]["train"][0][field]
    with pytest.raises(ValueError, match="prior question ID or context"):
        task.check_historical_exclusions(splits, controls())


def save_fixture(directory):
    cfg = config(train_size=4, validation_size=4, test_size=4)
    splits = {"train": rows("A", 1), "validation": rows("B", 1), "test": rows("C", 1)}
    data = directory / "data"
    data.mkdir()
    hashes = {}
    for name, values in splits.items():
        path = data / f"{name}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in values), encoding="utf-8")
        hashes[name] = task.sha256(path.read_bytes())
    manifest = directory / "split_manifest.json"
    manifest.write_text(json.dumps(task.public_manifest(splits)), encoding="utf-8")
    provenance = {"data_config": cfg, "local_data_sha256": hashes,
                  "split_manifest_sha256": task.sha256(manifest.read_bytes()), "historical_exclusion_controls": controls(),
                  "prior_manifest_sha256": [r["sha256"] for r in controls()["prior_manifests"]]}
    (directory / "data_provenance.json").write_text(json.dumps(provenance), encoding="utf-8")
    return cfg, splits


def test_prepared_reload_hashes_and_extra_configuration(tmp_path):
    cfg, expected = save_fixture(tmp_path)
    loaded, _ = task.load_prepared(tmp_path, {**cfg, "verifier_l2": 0.1, "some_other_runtime_option": True})
    assert loaded == expected
    with pytest.raises(ValueError, match="configuration differs"):
        task.load_prepared(tmp_path, {**cfg, "seed": cfg["seed"] + 1})
    (tmp_path / "data/test.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="local data checksum"):
        task.load_prepared(tmp_path, cfg)


def test_reload_checks_exclusions_from_saved_control_lists(tmp_path):
    cfg, splits = save_fixture(tmp_path)
    path = tmp_path / "data_provenance.json"
    provenance = json.loads(path.read_bytes())
    provenance["historical_exclusion_controls"]["all_prior_context_sha256"].append(splits["train"][0]["context_sha256"])
    path.write_text(json.dumps(provenance), encoding="utf-8")
    with pytest.raises(ValueError, match="repeats a prior"):
        task.load_prepared(tmp_path, cfg)


def test_prepare_refuses_existing_artifacts_before_any_download(tmp_path):
    (tmp_path / "data").mkdir()
    with pytest.raises(FileExistsError, match="preserved"):
        task.prepare(config(), tmp_path, tmp_path / "cache", False, [])


def test_ordered_manifest_controls_require_complete_distinct_sources():
    with pytest.raises(ValueError, match="At least one"):
        task.exclusion_controls([], [])
    with pytest.raises(ValueError, match="Duplicate"):
        task.exclusion_controls(historical(), ["same", "same"])
    with pytest.raises(ValueError, match="incomplete"):
        task.exclusion_controls([{"train": []}], ["missing"])
