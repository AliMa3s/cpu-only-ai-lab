import json
import unittest

from cpu_ai_lab.merge_benchmark import merge_schedule, observed_quality_check, summarize_merge


class MergeMethodTests(unittest.TestCase):
    def test_summary_labels_both_variants_as_merge_states(self):
        records = [
            {"variant": "unmerged", "pair": 0, "seconds": 2, "output_tokens": 32},
            {"variant": "merged", "pair": 0, "seconds": 1, "output_tokens": 32},
        ]
        summary = summarize_merge(records)
        self.assertEqual(summary["merged_speedup_ratio"], 2)
        self.assertNotIn("int8", json.dumps(summary))
        schedule = merge_schedule(2, 3, 42)
        self.assertEqual(sum(row["order"][0] == "merged" for row in schedule), 3)
        self.assertNotIn("int8", json.dumps(schedule))

    def test_quality_check_catches_roundoff_or_accuracy_regressions(self):
        baseline = {"loss": {"loss": 0.001}, "generation": {"json_exact_match_rate": 1.0, "valid_json_rate": 1.0}}
        self.assertTrue(observed_quality_check(baseline, baseline, 1e-5)["passed"])
        self.assertFalse(observed_quality_check(baseline, baseline, 0.1)["passed"])
        candidate = {"loss": {"loss": 0.001}, "generation": {"json_exact_match_rate": 0.5, "valid_json_rate": 1.0}}
        self.assertFalse(observed_quality_check(baseline, candidate, 1e-5)["passed"])


if __name__ == "__main__":
    unittest.main()
