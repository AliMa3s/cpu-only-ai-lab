"""Backend comparison accounting tests; no models or Ollama needed."""
import unittest
import json
from pathlib import Path
import tempfile

from cpu_ai_lab.backend_benchmark import check_files, make_schedule, quality_summary, reset_ollama_runner, sha256, summarize_pairs, verify_provenance


class BackendBenchmarkTests(unittest.TestCase):
    def test_timed_runner_reset_unloads_then_loads_without_prompt(self):
        class Client:
            loaded = True
            calls = []

            def request(self, path, payload=None):
                self.calls.append((path, payload))
                if path == "/api/ps":
                    return {"models": [{"name": "test", "digest": "sha", "size_vram": 0}] if self.loaded else []}
                self.loaded = payload["keep_alive"] != 0
                return {"done": True, "load_duration": 5}

        client = Client()
        result = reset_ollama_runner(client, "test", "sha", 4, 2048)
        self.assertEqual([path for path, _ in client.calls], ["/api/generate", "/api/ps", "/api/generate", "/api/ps"])
        load_payload = client.calls[2][1]
        self.assertNotIn("prompt", load_payload)
        self.assertEqual(load_payload["options"]["num_gpu"], 0)
        self.assertEqual(load_payload["options"]["num_thread"], 4)
        self.assertTrue(result["first_request_after_reload"])
        self.assertEqual(result["size_vram"], 0)

    def test_failed_unload_blocks_timed_run(self):
        class Client:
            def request(self, path, payload=None):
                return {"models": [{"name": "test"}]} if path == "/api/ps" else {}

        with self.assertRaisesRegex(RuntimeError, "remained loaded"):
            reset_ollama_runner(Client(), "test", "sha", 4, 2048)

    def test_schedule_balances_order_and_covers_each_pair_once(self):
        schedule = make_schedule(2, 3, 7)
        self.assertEqual(schedule, make_schedule(2, 3, 7))
        self.assertEqual(len({p["pair_id"] for p in schedule}), 6)
        self.assertEqual(sum(p["order"][0] == "pytorch_fp32" for p in schedule), 3)
        self.assertEqual(sum(p["order"][0] == "ollama_f16" for p in schedule), 3)

    def test_short_ollama_output_is_excluded_from_speed_claim(self):
        records = [
            {"pair_id": "a", "variant": "pytorch_fp32", "output_tokens": 32, "wall_seconds": 4, "output": "a"},
            {"pair_id": "a", "variant": "ollama_f16", "output_tokens": 32, "wall_seconds": 2, "output": "b"},
            {"pair_id": "b", "variant": "pytorch_fp32", "output_tokens": 32, "wall_seconds": 5, "output": "a"},
            {"pair_id": "b", "variant": "ollama_f16", "output_tokens": 8, "wall_seconds": 0.1, "output": "b"},
        ]
        result = summarize_pairs(records, 32)
        self.assertEqual(result["complete_fixed_length_pairs"], 1)
        self.assertEqual(result["median_paired_speedup_ratio"], 2)
        self.assertEqual(len(result["excluded_pairs"]), 1)
        self.assertEqual(result["ollama_aggregate_tokens_per_second"], 16)

    def test_no_speedup_reported_without_fixed_length_pairs(self):
        result = summarize_pairs([
            {"pair_id": "a", "variant": "pytorch_fp32", "output_tokens": 32, "wall_seconds": 4, "output": "a"},
            {"pair_id": "a", "variant": "ollama_f16", "output_tokens": 8, "wall_seconds": 0.1, "output": "b"},
        ], 32)
        self.assertEqual(result["status"], "no_comparable_pairs")
        self.assertNotIn("median_paired_speedup_ratio", result)

    def test_material_prompt_count_mismatch_excluded(self):
        result = summarize_pairs([
            {"pair_id": "a", "variant": "pytorch_fp32", "output_tokens": 32, "wall_seconds": 4, "output": "a"},
            {"pair_id": "a", "variant": "ollama_f16", "output_tokens": 32, "wall_seconds": 0.1, "output": "b",
             "hf_prompt_tokens": 60, "ollama_prompt_eval_count": 30, "prompt_counts_equal": False},
        ], 32)
        self.assertEqual(result["status"], "no_comparable_pairs")
        self.assertIn("prompt token", result["excluded_pairs"][0]["reason"])

    def test_duplicate_backend_in_pair_rejected(self):
        row = {"pair_id": "a", "variant": "pytorch_fp32", "output_tokens": 32, "wall_seconds": 4, "output": "a"}
        with self.assertRaises(ValueError):
            summarize_pairs([row, row], 32)

    def test_quality_summary_keeps_backend_rates_separate(self):
        rows = [
            {"variant": "pytorch_fp32", "json_exact": True, "valid_json": True, "text_exact": True},
            {"variant": "ollama_f16", "json_exact": False, "valid_json": True, "text_exact": False},
        ]
        result = quality_summary(rows)
        self.assertEqual(result["pytorch_fp32"]["json_exact_match_rate"], 1)
        self.assertEqual(result["ollama_f16"]["json_exact_match_rate"], 0)
        self.assertEqual(result["json_exact_match_drop"], 1)

    def test_artifact_corruption_and_path_escape_are_rejected(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            directory = Path(temporary)
            artifact = directory / "adapter.safetensors"
            artifact.write_bytes(b"test checkpoint bytes")
            digest = sha256(artifact)
            check_files(directory, {artifact.name: digest})
            artifact.write_bytes(b"modified checkpoint bytes")
            with self.assertRaises(ValueError):
                check_files(directory, {artifact.name: digest})
            with self.assertRaises(ValueError):
                check_files(directory, {"../outside": digest})

    def test_provenance_checks_gguf_and_training_data(self):
        with tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent) as temporary:
            root = Path(temporary)
            training, deployment = root / "train", root / "deploy"
            for directory in (training / "adapter", training / "data", deployment / "model"):
                directory.mkdir(parents=True)
            def save(path, value):
                path.write_text(json.dumps(value), encoding="utf-8")
            save(training / "resolved_config.json", {"model_id": "model", "revision": "pin"})
            save(training / "run_status.json", {"status": "complete"})
            (training / "adapter" / "adapter.safetensors").write_bytes(b"adapter")
            (training / "data" / "test.jsonl").write_bytes(b"{}\n")
            (deployment / "model" / "model.safetensors").write_bytes(b"float16")
            (deployment / "model.gguf").write_bytes(b"gguf")
            adapter_hash = {"adapter.safetensors": sha256(training / "adapter" / "adapter.safetensors")}
            save(training / "metrics.json", {"artifact_sha256": adapter_hash,
                                             "dataset": {"sha256": {"test": sha256(training / "data" / "test.jsonl")}}})
            manifest = {
                "format": "cpu-ai-lab-ollama-deployment-v1", "ollama_created": True,
                "dtype": "float16", "merge_dtype": "float32", "model_id": "model", "model_revision": "pin",
                "training_metrics_sha256": sha256(training / "metrics.json"),
                "adapter_directory": "../train/adapter", "adapter_sha256": adapter_hash,
                "model_directory": "model", "model_files_sha256": {"model.safetensors": sha256(deployment / "model" / "model.safetensors")},
                "ollama_model": "model:f16", "ollama_digest": "digest", "import_format": "gguf",
                "gguf_file": "model.gguf", "gguf_sha256": sha256(deployment / "model.gguf"),
            }
            save(deployment / "manifest.json", manifest)
            self.assertEqual(verify_provenance(training, deployment / "manifest.json")[2]["ollama_digest"], "digest")
            (deployment / "model.gguf").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "integrity"):
                verify_provenance(training, deployment / "manifest.json")


if __name__ == "__main__":
    unittest.main()
