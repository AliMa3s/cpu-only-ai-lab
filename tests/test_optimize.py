import unittest

from cpu_ai_lab.optimize import choose_subset, make_schedule, quality_gate, select_linear_names, select_variant, summarize_timings


def quality(loss=1.0, exact=0.5, valid=0.75):
    return {"loss": {"loss": loss}, "generation": {"json_exact_match_rate": exact, "valid_json_rate": valid}}


class OptimizationMethodTests(unittest.TestCase):
    def test_schedule_pairs_every_prompt_and_balances_order(self):
        schedule = make_schedule(2, 3, 17)
        self.assertEqual(len(schedule), 6)
        self.assertEqual({(row["prompt_index"], row["repeat"]) for row in schedule}, {(p, r) for p in range(2) for r in range(3)})
        self.assertEqual(sum(row["order"][0] == "int8" for row in schedule), 3)
        self.assertEqual(schedule, make_schedule(2, 3, 17))

    def test_quality_gate_does_not_hide_accuracy_regression(self):
        self.assertFalse(quality_gate(quality(), quality(loss=0.8, exact=0.25))["passed"])
        self.assertFalse(quality_gate(quality(), quality(loss=1.2))["passed"])
        self.assertTrue(quality_gate(quality(), quality(loss=1.02))["passed"])

    def test_nonfinite_loss_fails_guard(self):
        result = quality_gate(quality(), quality(loss=float("nan")))
        self.assertFalse(result["passed"])
        self.assertIsNone(result["observed_loss_ratio"])

    def test_absolute_loss_tolerance_is_explicit_and_does_not_relax_accuracy(self):
        baseline = quality(loss=0.0003, exact=1.0, valid=1.0)
        candidate = quality(loss=0.005, exact=1.0, valid=1.0)
        self.assertFalse(quality_gate(baseline, candidate)["passed"])
        check = quality_gate(baseline, candidate, max_loss_increase=0.01)
        self.assertTrue(check["passed"])
        self.assertAlmostEqual(check["allowed_candidate_loss"], 0.0103)
        candidate["generation"]["json_exact_match_rate"] = 0.99
        self.assertFalse(quality_gate(baseline, candidate, max_loss_increase=0.01)["passed"])

    def test_mlp_scope_excludes_attention_and_output_head(self):
        names = ["model.layers.0.mlp.gate_proj", "model.layers.0.mlp.up_proj", "model.layers.0.mlp.down_proj", "model.layers.0.self_attn.q_proj", "lm_head"]
        self.assertEqual(select_linear_names(names, "mlp-only"), names[:3])
        self.assertEqual(select_linear_names(names, "all-linear"), names)
        with self.assertRaises(ValueError):
            select_linear_names(names, "unknown")

    def test_timing_uses_total_time_and_rejects_unmatched_work(self):
        records = [
            {"pair": 0, "variant": "fp32", "seconds": 4.0, "output_tokens": 32},
            {"pair": 0, "variant": "int8", "seconds": 2.0, "output_tokens": 32},
            {"pair": 1, "variant": "fp32", "seconds": 6.0, "output_tokens": 32},
            {"pair": 1, "variant": "int8", "seconds": 3.0, "output_tokens": 32},
        ]
        summary = summarize_timings(records)
        self.assertEqual(summary["fp32"]["aggregate_tokens_per_second"], 6.4)
        self.assertEqual(summary["int8_speedup_ratio"], 2.0)
        records[-1]["output_tokens"] = 31
        with self.assertRaises(ValueError):
            summarize_timings(records)

    def test_fast_but_bad_quality_is_not_selected(self):
        timing = {"int8_speedup_ratio": 2.0, "paired_min_speedup_ratio": 1.5}
        self.assertEqual(select_variant({"passed": False}, timing)["variant"], "fp32")
        self.assertEqual(select_variant({"passed": True}, timing)["variant"], "int8")
        timing["paired_min_speedup_ratio"] = 0.9
        self.assertEqual(select_variant({"passed": True}, timing)["variant"], "fp32")

    def test_subsets_are_deterministic_and_without_replacement(self):
        subset = choose_subset(list(range(20)), 8, 42)
        self.assertEqual(len(subset), len(set(subset)))
        self.assertEqual(subset, choose_subset(list(range(20)), 8, 42))
        self.assertEqual(choose_subset([1, 2], 8, 42), [1, 2])


if __name__ == "__main__":
    unittest.main()
