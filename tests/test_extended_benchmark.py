"""Long-output measurement accounting; no model downloads or model execution."""
import copy
import json
from pathlib import Path
import unittest

from cpu_ai_lab.extended_benchmark import MemorySampler, make_schedule, ollama_process_tree, select_threads, summarize, validate_extended_config


class ExtendedBenchmarkTests(unittest.TestCase):
    def config(self):
        return json.loads((Path(__file__).resolve().parents[1] / "configs" / "extended-benchmark.json").read_text())

    def pair(self, pair_id="p", budget=128, pytorch_tokens=128, ollama_tokens=128):
        return [
            {"pair_id": pair_id, "prompt_id": "forest", "variant": "pytorch_fp32", "requested_tokens": budget,
             "output_tokens": pytorch_tokens, "prompt_tokens": 50, "output": "sample", "wall_seconds": 8,
             "first_content_seconds": 0.2},
            {"pair_id": pair_id, "prompt_id": "forest", "variant": "ollama_f16", "requested_tokens": budget,
             "output_tokens": ollama_tokens, "prompt_tokens": 50, "output": "sample", "wall_seconds": 2,
             "first_content_seconds": 0.3, "prompt_cached_tokens": 0},
        ]

    def test_confirmation_prompts_cannot_leak_into_tuning(self):
        cfg = self.config()
        validate_extended_config(cfg)
        cfg["confirmation_prompts"][0]["text"] = cfg["tuning_prompts"][0]["text"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            validate_extended_config(cfg)

    def test_schedule_covers_all_budgets_and_balances_backend_order(self):
        cfg = self.config()
        schedule = make_schedule(cfg["confirmation_prompts"], [128, 256], 3, 9)
        self.assertEqual(len(schedule), 24)
        self.assertEqual(len({r["pair_id"] for r in schedule}), 24)
        self.assertEqual(sum(r["order"][0] == "pytorch_fp32" for r in schedule), 12)
        self.assertEqual(schedule, make_schedule(cfg["confirmation_prompts"], [128, 256], 3, 9))

    def test_threads_selected_only_after_same_work_is_confirmed(self):
        rows = [{"prompt_id": p, "repeat": 1, "threads": t, "output_tokens": 64,
                 "output": "identical", "wall_seconds": seconds}
                for p in ("a", "b") for t, seconds in ((1, 1.0), (4, 2.0))]
        self.assertEqual(select_threads(rows, [1, 4], 2)["selected_threads"], 1)
        changed = copy.deepcopy(rows)
        changed[0]["output_tokens"] = 20
        with self.assertRaisesRegex(ValueError, "changed generated work"):
            select_threads(changed, [1, 4], 2)
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            select_threads(rows[:-1], [1, 4], 2)

    def test_early_stops_retained_without_entering_fixed_work_summary(self):
        rows = self.pair() + self.pair("short", ollama_tokens=30)
        summary = summarize(rows)
        self.assertEqual(summary["fixed_work"]["pairs"], 1)
        self.assertEqual(summary["fixed_work"]["median_paired_speedup"], 4)
        self.assertEqual(len(summary["unmatched_or_early_stop_pairs"]), 1)
        self.assertEqual(summary["all_requests"]["ollama_f16"]["requests"], 2)
        self.assertEqual(summary["all_requests"]["ollama_f16"]["natural_early_stops"], 1)

    def test_even_one_prompt_token_mismatch_disqualifies_fixed_work(self):
        rows = self.pair()
        rows[1]["prompt_tokens"] = 49
        summary = summarize(rows)
        self.assertEqual(summary["status"], "no_exact_budget_pairs")
        self.assertEqual(summary["fixed_work"]["pairs"], 0)

    def test_cached_prompt_disqualifies_fixed_work(self):
        rows = self.pair()
        rows[1]["prompt_cached_tokens"] = 10
        summary = summarize(rows)
        self.assertEqual(summary["fixed_work"]["pairs"], 0)
        self.assertIn("cached", summary["unmatched_or_early_stop_pairs"][0]["reasons"][0])

    def test_invalid_or_duplicate_timing_cannot_create_speed_claim(self):
        rows = self.pair()
        rows[0]["wall_seconds"] = float("nan")
        with self.assertRaisesRegex(ValueError, "Invalid timing"):
            summarize(rows)
        rows = self.pair()
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            summarize(rows + [rows[0]])

    def test_budget_results_are_reported_separately(self):
        rows = self.pair() + self.pair("long", 256, 256, 256)
        summary = summarize(rows)
        self.assertEqual(summary["by_token_budget"]["128"]["pairs"], 1)
        self.assertEqual(summary["by_token_budget"]["256"]["pairs"], 1)
        self.assertEqual(summary["fixed_work"]["pairs"], 2)

    def test_memory_includes_native_runner_descendants_and_limits_metadata(self):
        snapshot = [
            {"pid": 10, "ppid": 1, "name": "ollama app.exe", "rss_bytes": 100, "cmdline": "must not be saved"},
            {"pid": 11, "ppid": 10, "name": "ollama.exe", "rss_bytes": 200},
            {"pid": 12, "ppid": 11, "name": "llama-server.exe", "rss_bytes": 900},
            {"pid": 13, "ppid": 12, "name": "helper.exe", "rss_bytes": 30},
            {"pid": 99, "ppid": 1, "name": "unrelated.exe", "rss_bytes": 5000},
        ]
        selected = ollama_process_tree(snapshot)
        self.assertEqual([row["pid"] for row in selected], [10, 11, 12, 13])
        self.assertEqual(sum(row["rss_bytes"] for row in selected), 1230)
        self.assertTrue(all(set(row) == {"pid", "ppid", "name", "rss_bytes"} for row in selected))

    def test_memory_rediscovers_runner_started_after_first_sample(self):
        from types import SimpleNamespace
        sampler = MemorySampler()
        sampler.process = SimpleNamespace(memory_info=lambda: SimpleNamespace(rss=50))
        initial = [{"pid": 10, "ppid": 1, "name": "ollama.exe", "rss_bytes": 100}]
        later = initial + [{"pid": 11, "ppid": 10, "name": "llama-server.exe", "rss_bytes": 900}]
        snapshots = iter((initial, later))
        sampler._read_process_snapshot = lambda: next(snapshots)
        sampler._sample_once()
        self.assertEqual(sampler.result["ollama_peak_rss_sum_bytes"], 100)
        sampler._sample_once()
        self.assertEqual(sampler.result["ollama_peak_rss_sum_bytes"], 1000)
        self.assertIn("llama-server.exe", sampler.result["ollama_process_names"])
        self.assertEqual(sampler.result["samples"], 2)


if __name__ == "__main__":
    unittest.main()
