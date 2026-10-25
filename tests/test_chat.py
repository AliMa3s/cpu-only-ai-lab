"""Test CPU settings and benchmark decisions without running a model."""
import copy
import unittest
from unittest.mock import patch

from cpu_ai_lab.chat import generate_cpu, select_threads


def benchmark_report():
    return {
        "status": "complete", "selected_threads": 4,
        "config": {"model": "model", "context": 2048, "baseline_threads": 12},
        "model": {"digest": "sha"},
        "confirmation_summary": {
            "status": "measured", "candidate_threads": 4, "baseline_threads": 12,
            "complete_pairs": 2, "candidate_faster_pairs": 2,
            "median_paired_speedup": 1.2, "all_output_lengths_equal": True,
            "pairs": [
                {"speedup": 1.1, "baseline_tokens": 64, "candidate_tokens": 64},
                {"speedup": 1.3, "baseline_tokens": 64, "candidate_tokens": 64},
            ],
        },
    }


class ChatTests(unittest.TestCase):
    def test_default_and_explicit_threads(self):
        self.assertEqual(select_threads(None, "model", "sha", 2048)[0], 12)
        self.assertEqual(select_threads(benchmark_report(), "model", "sha", 2048, 6)[0], 6)
        with self.assertRaises(ValueError):
            select_threads(None, "model", "sha", 2048, 0)

    def test_confirmed_same_model_candidate_is_used(self):
        self.assertEqual(select_threads(benchmark_report(), "model", "sha", 2048)[0], 4)

    def test_mismatched_model_or_context_rejects_candidate(self):
        report = benchmark_report()
        self.assertEqual(select_threads(report, "other", "sha", 2048)[0], 12)
        self.assertEqual(select_threads(report, "model", "other-sha", 2048)[0], 12)
        self.assertEqual(select_threads(report, "model", "sha", 4096)[0], 12)

    def test_tuning_only_incomplete_and_mixed_results_reject_candidate(self):
        report = benchmark_report()
        for key, value in (("status", "running"), ("confirmation_summary", {})):
            changed = copy.deepcopy(report)
            changed[key] = value
            self.assertEqual(select_threads(changed, "model", "sha", 2048)[0], 12)
        for key, value in (("candidate_faster_pairs", 1), ("all_output_lengths_equal", False),
                           ("median_paired_speedup", 0.9), ("pairs", [])):
            changed = copy.deepcopy(report)
            changed["confirmation_summary"][key] = value
            self.assertEqual(select_threads(changed, "model", "sha", 2048)[0], 12)

    def test_generate_forces_cpu_and_leaves_model_loaded(self):
        class FakeClient:
            def __init__(self):
                self.calls = []

            def request(self, path, payload=None):
                self.calls.append((path, payload))
                if path == "/api/tags":
                    return {"models": [{"name": "model", "digest": "sha"}]}
                if path == "/api/ps":
                    return {"models": [{"name": "model", "digest": "sha", "size_vram": 0}]}
                raise AssertionError("Unexpected request, including an unload")

            def generate(self, payload):
                self.calls.append(("generate", payload))
                return ({"eval_count": 10, "eval_duration": 2_000_000_000}, 2.1, 0.1, "answer", "")

        fake = FakeClient()
        with patch("cpu_ai_lab.chat.OllamaClient", return_value=fake):
            answer, audit = generate_cpu("question", model="model", benchmark=benchmark_report())
        self.assertEqual(answer, "answer")
        self.assertEqual(audit["requested_threads"], 4)
        self.assertEqual(audit["cpu_evidence"]["reported_vram_bytes"], 0)
        payload = next(payload for path, payload in fake.calls if path == "generate")
        self.assertEqual(payload["options"]["num_gpu"], 0)
        self.assertEqual(payload["keep_alive"], "2m")
        self.assertFalse(any(path == "/api/generate" for path, _ in fake.calls))


if __name__ == "__main__":
    unittest.main()
