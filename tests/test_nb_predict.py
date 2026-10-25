"""Standalone classifier must agree with training and avoid model-framework imports."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import pytest

from cpu_ai_lab import nb_predict
from cpu_ai_lab.real_task import naive_bayes, normalize_template, predict_naive_bayes


def row(index, text, label):
    return {"id": index, "text": text, "label": label, "text_sha256": str(index)}


def create_recipe(path):
    splits = {
        "train": [row(1, "lunch meet friend", 0), row(2, "free prize winner", 1)],
        "validation": [row(3, "friend lunch", 0), row(4, "prize winner", 1)],
        "test": [row(5, "friend lunch secretword", 0), row(6, "free prize secretword", 1)],
    }
    recipe = path / "naive_bayes.json"
    naive_bayes(splits, recipe_output=recipe)
    (path / "run_status.json").write_text(json.dumps({"status": "complete"}))
    (path / "metrics.json").write_text(json.dumps({"naive_bayes_recipe_sha256": hashlib.sha256(recipe.read_bytes()).hexdigest()}))


def test_standalone_predictions_match_original_classifier():
    with tempfile.TemporaryDirectory() as name:
        path = Path(name)
        create_recipe(path)
        recipe = nb_predict.load_recipe(path)
        for message in ("free prize", "meet friend", "unknown vocabulary", "", "lunch lunch free winner", "FREE! Call 12345"):
            prediction = nb_predict.predict_text(recipe, message)
            assert prediction["label"] == predict_naive_bayes(path, message)["label"]
            assert prediction["scores_are_probabilities"] is False
            assert set(prediction["log_scores"]) == {"ham", "spam"}


def test_normalization_matches_training_for_contacts_and_unicode():
    for message in ("FREE!!! prize123 call 456", "Visit https://example.test/x and WWW.site.test", "Mail a.b+tag@domain.example", "Café ＦＲＥＥ １２３", "Hi\nHow's lunch?", ""):
        assert nb_predict.normalize_template(message) == normalize_template(message)


def test_hash_and_completion_checks_precede_prediction():
    with tempfile.TemporaryDirectory() as name:
        path = Path(name)
        create_recipe(path)
        (path / "naive_bayes.json").write_text("{}")
        with pytest.raises(ValueError, match="SHA256"):
            nb_predict.load_recipe(path)
        (path / "run_status.json").write_text(json.dumps({"status": "running"}))
        with pytest.raises(ValueError, match="completed"):
            nb_predict.load_recipe(path)


def test_import_in_clean_process_loads_no_model_framework():
    code = "import sys; import cpu_ai_lab.nb_predict; assert not any(name.split('.')[0] in {'torch', 'transformers', 'peft', 'numpy'} for name in sys.modules); print('stdlib-only')"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True, timeout=10)
    assert result.stdout.strip() == "stdlib-only"


def test_malformed_recipe_cannot_use_invalid_smoothing_or_vocabulary():
    with tempfile.TemporaryDirectory() as name:
        path = Path(name)
        create_recipe(path)
        recipe = nb_predict.load_recipe(path)
        recipe["alpha"] = float("nan")
        with pytest.raises(ValueError, match="finite"):
            nb_predict.validate_recipe(recipe)
