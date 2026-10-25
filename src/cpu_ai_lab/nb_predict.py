"""Use the saved CPU SMS Naive Bayes classifier with Python's standard library.

This command imports no tensor or model framework. It verifies a completed run
and the recorded recipe checksum before scoring. Its log-scores are comparison
values, not calibrated probabilities.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import unicodedata


def normalize_template(text: str) -> str:
    """The same v1 normalization used in real_task, without importing its stack."""
    text = unicodedata.normalize("NFKC", text).casefold()
    text = re.sub(r"(?:https?://|www\.)\S+", " url ", text)
    text = re.sub(r"[\w.+-]+@[\w.-]+\.[a-z]{2,}", " email ", text)
    text = re.sub(r"\d+", " number ", text)
    return " ".join(re.findall(r"[a-z]+", text))


def validate_recipe(recipe: dict) -> None:
    if recipe.get("normalization") != "normalize_template v1":
        raise ValueError("Unsupported Naive Bayes text normalization.")
    alpha = recipe.get("alpha")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha) or alpha <= 0:
        raise ValueError("Naive Bayes alpha must be finite and positive.")
    documents = recipe.get("class_document_counts", {})
    counts = recipe.get("word_counts", {})
    if set(documents) != {"0", "1"} or set(counts) != {"0", "1"}:
        raise ValueError("The saved classifier must contain both binary classes.")
    for label in ("0", "1"):
        if type(documents[label]) is not int or documents[label] < 1:
            raise ValueError("Class document counts must be positive integers.")
        if not isinstance(counts[label], dict) or not counts[label]:
            raise ValueError("Each class must have a nonempty token vocabulary.")
        for word, count in counts[label].items():
            if not isinstance(word, str) or not word or type(count) is not int or count < 1:
                raise ValueError("Word counts must map nonempty tokens to positive integers.")
    vocabulary = set(counts["0"]) | set(counts["1"])
    if recipe.get("vocabulary_size") != len(vocabulary):
        raise ValueError("Saved vocabulary size does not match the word counts.")


def load_recipe(run: Path) -> dict:
    status = json.loads((run / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError("Only completed training runs can be loaded.")
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    payload = (run / "naive_bayes.json").read_bytes()
    if hashlib.sha256(payload).hexdigest() != metrics.get("naive_bayes_recipe_sha256"):
        raise ValueError("Naive Bayes recipe failed its recorded SHA256 check.")
    recipe = json.loads(payload)
    validate_recipe(recipe)
    return recipe


def predict_text(recipe: dict, text: str) -> dict:
    counts = recipe["word_counts"]
    vocabulary = set(counts["0"]) | set(counts["1"])
    words = Counter(word for word in normalize_template(text).split() if word in vocabulary)
    documents = recipe["class_document_counts"]
    scores = {}
    for label, name in (("0", "ham"), ("1", "spam")):
        denominator = sum(counts[label].values()) + recipe["alpha"] * len(vocabulary)
        score = math.log(documents[label] / sum(documents.values()))
        score += sum(number * math.log((counts[label].get(word, 0) + recipe["alpha"]) / denominator) for word, number in words.items())
        scores[name] = score
    return {
        "label": "spam" if scores["spam"] > scores["ham"] else "ham",
        "log_scores": scores,
        "spam_minus_ham_log_score": scores["spam"] - scores["ham"],
        "scores_are_probabilities": False,
        "score_description": "Unnormalized Naive Bayes log-scores; larger wins. These are not calibrated probabilities.",
        "matched_vocabulary_token_count": sum(words.values()),
        "method": "multinomial_naive_bayes", "device": "cpu", "runtime": "Python standard library only",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True, help="Completed real_task experiment directory")
    parser.add_argument("--text", required=True, help="SMS text to classify")
    args = parser.parse_args()
    print(json.dumps(predict_text(load_recipe(args.run), args.text), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
