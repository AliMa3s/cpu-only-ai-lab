"""Metric, CPU-proof and experimental-separation tests; no Ollama needed."""
import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

from cpu_ai_lab.inference import cpu_status, paired_summary, response_metrics, run_experiment, validate_config


class FakeOllama:
    def __init__(self, *_):
        self.loaded = False
        self.payloads = []

    def request(self, path, payload=None):
        if path == "/api/ps":
            return {"models": [{"name": "qwen2.5-coder:7b", "digest": "immutable-sha", "size_vram": 0}]
                    if self.loaded else []}
        if path == "/api/tags":
            return {"models": [{"name": "qwen2.5-coder:7b", "digest": "immutable-sha"}]}
        if path == "/api/version":
            return {"version": "fake-for-test"}
        if path == "/api/generate" and payload["keep_alive"] == 0:
            self.loaded = False
            return {"done": True}
        raise AssertionError(path)

    def generate(self, payload):
        self.loaded = True
        self.payloads.append(payload)
        count = payload["options"]["num_predict"]
        tps = 10 if payload["options"]["num_thread"] == 4 else 5
        return ({"done": True, "eval_count": count, "eval_duration": int(count / tps * 1e9)},
                count / tps + 0.1, 0.1, "same generated text", "")


class InferenceTests(unittest.TestCase):
    def test_cpu_status_requires_explicit_zero_and_matching_digest(self):
        valid = {"models": [{"name": "test", "size_vram": 0, "digest": "sha"}]}
        self.assertEqual(cpu_status(valid, "test", "sha")["size_vram"], 0)
        for vram in (None, 1, False, "0"):
            with self.assertRaises(RuntimeError):
                cpu_status({"models": [{"name": "test", "size_vram": vram, "digest": "sha"}]}, "test", "sha")
        with self.assertRaises(RuntimeError):
            cpu_status(valid, "test", "different")

    def test_metrics_use_decode_time_not_total_request_time(self):
        metrics = response_metrics({"eval_count": 64, "eval_duration": 8_000_000_000,
                                    "total_duration": 20_000_000_000}, 21, 12)
        self.assertEqual(metrics["decode_tokens_per_second"], 8)
        self.assertEqual(metrics["request_wall_seconds"], 21)
        self.assertIsNone(metrics["prompt_cached_tokens"])
        with self.assertRaises(RuntimeError):
            response_metrics({"eval_count": 0, "eval_duration": 1}, 1, None)

    def test_tuning_and_confirmation_prompts_cannot_overlap(self):
        with self.assertRaises(ValueError):
            validate_config({"tuning_prompts": ["same"], "confirmation_prompts": ["same"]})
        with self.assertRaises(ValueError):
            validate_config({"base_url": "https://remote.example.com"})
        with self.assertRaises(ValueError):
            validate_config({"threads": [2, 2, 12]})
        self.assertEqual(validate_config({})["baseline_threads"], 12)

    def test_paired_summary_matches_pairs_and_ignores_incomplete_pair(self):
        rows = [
            {"pair_id": "a", "threads": 12, "decode_tokens_per_second": 4, "output_tokens": 64, "output": "one"},
            {"pair_id": "b", "threads": 4, "decode_tokens_per_second": 12, "output_tokens": 64, "output": "two"},
            {"pair_id": "b", "threads": 12, "decode_tokens_per_second": 6, "output_tokens": 64, "output": "two"},
            {"pair_id": "a", "threads": 4, "decode_tokens_per_second": 8, "output_tokens": 64, "output": "one"},
            {"pair_id": "c", "threads": 12, "decode_tokens_per_second": 100, "output_tokens": 64, "output": "ignored"},
        ]
        result = paired_summary(rows, 12, 4)
        self.assertEqual(result["complete_pairs"], 2)
        self.assertEqual(result["median_paired_speedup"], 2)
        self.assertEqual(result["improvement_percent"], 100)
        self.assertEqual(result["candidate_faster_pairs"], 2)
        self.assertTrue(result["all_output_lengths_equal"])

    def test_no_speedup_claim_if_baseline_selected(self):
        result = paired_summary([], 12, 12)
        self.assertEqual(result["speedup"], 1)
        self.assertEqual(result["improvement_percent"], 0)

    def test_full_experiment_uses_heldout_pairs_and_unloads_only_test_model(self):
        fake = FakeOllama()
        config = {"threads": [4, 12], "tuning_prompts": ["tuning input"],
                  "confirmation_prompts": ["heldout input one", "heldout input two"]}
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary, patch("cpu_ai_lab.inference.OllamaClient", return_value=fake), patch("builtins.print"):
            path = Path(temporary) / "result.json"
            report = run_experiment(config, path)
            saved = json.loads(path.read_text(encoding="utf-8"))
            self.assertEqual(saved["status"], "complete")
            self.assertEqual(report["selected_threads"], 4)
            self.assertEqual(report["confirmation_summary"]["complete_pairs"], 4)
            self.assertEqual(report["confirmation_summary"]["median_paired_speedup"], 2)
            self.assertTrue(all(p["options"]["num_gpu"] == 0 for p in fake.payloads))
            self.assertFalse(fake.loaded)
            self.assertEqual({r["prompt"] for r in report["tuning"]}, {"tuning input"})
            self.assertEqual({r["prompt"] for r in report["confirmation"]}, {"heldout input one", "heldout input two"})
            with self.assertRaises(FileExistsError):
                run_experiment(config, path)

    def test_preexisting_loaded_model_is_not_unloaded(self):
        fake = FakeOllama()
        fake.loaded = True
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary, patch("cpu_ai_lab.inference.OllamaClient", return_value=fake):
            with self.assertRaisesRegex(RuntimeError, "already has a model"):
                run_experiment({}, Path(temporary) / "result.json")
            self.assertTrue(fake.loaded)
            self.assertEqual(fake.payloads, [])


if __name__ == "__main__":
    unittest.main()
