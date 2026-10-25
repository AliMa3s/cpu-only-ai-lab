"""Measure local CPU Ollama process-tree memory, independently of speed trials.

This corrective follow-up does not load a PyTorch model. It uses one raw prompt
and records memory during a natural, capped generation with verified zero VRAM.
"""
from __future__ import annotations

import argparse
import datetime as dt
from pathlib import Path
import json

from .backend_benchmark import load_json, reset_ollama_runner, sha256, verify_provenance
from .extended_benchmark import MemorySampler, validate_extended_config
from .inference import OllamaClient, atomic_write, cpu_status


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError("Choose a new memory report path; existing measurements are preserved")
    cfg = validate_extended_config(load_json(args.config))
    source_cfg, training_metrics, manifest = verify_provenance(args.training_dir, args.deployment_manifest)
    client = OllamaClient(cfg["base_url"], 300)
    if client.request("/api/ps").get("models"):
        raise RuntimeError("Ollama must have no models loaded before this memory-only probe")
    name, digest = manifest["ollama_model"], manifest["ollama_digest"]
    tags = [m for m in client.request("/api/tags").get("models", []) if m.get("name") == name]
    if len(tags) != 1 or tags[0].get("digest") != digest or tags[0].get("details", {}).get("quantization_level") != "F16":
        raise ValueError("Installed F16 model does not match the deployment manifest")
    prompt = cfg["confirmation_prompts"][0]
    report = {
        "format": "cpu-ai-lab-ollama-memory-probe-v1", "status": "running",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "purpose": "Corrective process-tree memory measurement; not a speed comparison",
        "provenance": {"model_id": source_cfg["model_id"], "revision": source_cfg["revision"],
                       "adapter_sha256": training_metrics["artifact_sha256"], "ollama_digest": digest,
                       "gguf_sha256": manifest.get("gguf_sha256"), "deployment_manifest_sha256": sha256(args.deployment_manifest),
                       "benchmark_config_sha256": sha256(args.config)},
        "ollama_version": client.request("/api/version"),
        "workload": {"model": name, "prompt_id": prompt["id"], "prompt": prompt["text"],
                     "raw": True, "requested_tokens": 128, "threads": cfg["ollama_threads"], "context": cfg["context"]},
        "limits": [
            "One memory-only follow-up request; no wall-time or throughput comparison is claimed.",
            "No PyTorch model is loaded in this process, unlike the original paired benchmark.",
            "RSS is sampled every 0.1 seconds and may miss short peaks; process sums can count shared pages more than once.",
            "Ollama API size is a server-reported model allocation, not an independently measured process RSS or whole-machine total.",
            "Process evidence contains only PID, parent PID, executable name and RSS; no command lines or environment values.",
        ],
    }
    atomic_write(args.output, report)
    started = False
    try:
        started = True
        report["cache_reset"] = reset_ollama_runner(client, name, digest, cfg["ollama_threads"], cfg["context"])
        before = cpu_status(client.request("/api/ps"), name, digest)
        payload = {"model": name, "prompt": prompt["text"], "raw": True, "keep_alive": "10m",
                   "options": {"num_gpu": 0, "num_thread": cfg["ollama_threads"], "num_ctx": cfg["context"],
                               "num_predict": 128, "temperature": 0, "seed": cfg["seed"], "repeat_penalty": 1.0, "stop": []}}
        with MemorySampler() as memory:
            response, _elapsed, _ttft, output, _thinking = client.generate(payload)
        after = cpu_status(client.request("/api/ps"), name, digest)
        report["memory"] = memory.result
        report["served_model_allocation"] = {
            "before": {"name": before.get("name"), "digest": before.get("digest"), "reported_size_bytes": before.get("size"), "size_vram": before["size_vram"]},
            "after": {"name": after.get("name"), "digest": after.get("digest"), "reported_size_bytes": after.get("size"), "size_vram": after["size_vram"]},
        }
        report["generation"] = {"output_tokens": response.get("eval_count"), "prompt_tokens": response.get("prompt_eval_count"),
                                "done_reason": response.get("done_reason"), "output": output}
        report["cpu_evidence"] = {"num_gpu": 0, "size_vram_before": before["size_vram"], "size_vram_after": after["size_vram"], "digest": digest}
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if started:
            try:
                client.request("/api/generate", {"model": name, "keep_alive": 0, "stream": False, "options": {"num_gpu": 0}})
                report["cleanup"] = "Test model unloaded" if not client.request("/api/ps").get("models") else "Other models remain loaded"
            except Exception as exc:
                report["cleanup"] = f"Unload failed: {exc}"
        report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_write(args.output, report)
    print(json.dumps({"status": report["status"], "ollama_peak_rss_sum_bytes": report["memory"]["ollama_peak_rss_sum_bytes"],
                      "process_names": report["memory"]["ollama_process_names"], "size_vram": report["cpu_evidence"]["size_vram_after"]}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/extended-benchmark.json"))
    parser.add_argument("--training-dir", type=Path, default=Path("artifacts/train-run"))
    parser.add_argument("--deployment-manifest", type=Path, default=Path("artifacts/deployment-f16/deployment-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/backend-memory.json"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
