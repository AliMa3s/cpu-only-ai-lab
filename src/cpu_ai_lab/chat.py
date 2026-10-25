"""Generate locally with a CPU-only Ollama model and optional measured settings.

Example: python -m cpu_ai_lab.chat --prompt "Explain merge sort" --benchmark results/inference.json
The model must already be installed. Answers go to stdout and audit data to stderr.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

from .inference import OllamaClient, cpu_status, response_metrics, validate_config


def select_threads(
    report: dict | None, model: str, digest: str, context: int,
    explicit_threads: int | None = None,
) -> tuple[int, str]:
    """Conservatively reuse a confirmed setting, never tuning-only winners."""
    if explicit_threads is not None:
        if type(explicit_threads) is not int or not 1 <= explicit_threads <= 256:
            raise ValueError("threads must be an integer in [1, 256]")
        return explicit_threads, "Explicit thread count; benchmark selection was not used."
    baseline = 12
    if report is None:
        return baseline, "Default 12 threads; no benchmark report supplied."
    if not isinstance(report, dict):
        return baseline, "Benchmark report is not an object; using 12 threads."
    cfg = report.get("config", {})
    installed = report.get("model", {})
    if not isinstance(cfg, dict) or not isinstance(installed, dict):
        return baseline, "Benchmark metadata is invalid; using 12 threads."
    if cfg.get("model") != model or installed.get("digest") != digest:
        return baseline, "Benchmark model or digest differs from the installed model; using 12 threads."
    if cfg.get("context") != context:
        return baseline, "Benchmark context differs from this request; using 12 threads."
    if type(cfg.get("baseline_threads")) is int and 1 <= cfg["baseline_threads"] <= 256:
        baseline = cfg["baseline_threads"]
    summary = report.get("confirmation_summary", {})
    candidate = report.get("selected_threads")
    if report.get("status") != "complete" or not isinstance(summary, dict) or summary.get("status") != "measured":
        return baseline, "No completed paired confirmation; using the benchmark baseline."
    pairs = summary.get("complete_pairs")
    ratio = summary.get("median_paired_speedup")
    if (
        type(candidate) is not int or not 1 <= candidate <= 256
        or summary.get("candidate_threads") != candidate
        or summary.get("baseline_threads") != baseline
        or type(pairs) is not int or pairs < 2
        or summary.get("candidate_faster_pairs") != pairs
        or not isinstance(ratio, (int, float)) or isinstance(ratio, bool)
        or not math.isfinite(ratio) or ratio <= 1
        or summary.get("all_output_lengths_equal") is not True
    ):
        return baseline, "Confirmation did not consistently favor a comparable candidate; using the baseline."
    raw_pairs = summary.get("pairs")
    if not isinstance(raw_pairs, list) or len(raw_pairs) != pairs:
        return baseline, "Confirmation pair records are missing; using the baseline."
    for pair in raw_pairs:
        if not isinstance(pair, dict):
            return baseline, "Invalid confirmation pair; using the baseline."
        speedup = pair.get("speedup")
        if (
            not isinstance(speedup, (int, float)) or isinstance(speedup, bool)
            or not math.isfinite(speedup) or speedup <= 1
            or type(pair.get("baseline_tokens")) is not int or pair["baseline_tokens"] <= 0
            or pair.get("candidate_tokens") != pair["baseline_tokens"]
        ):
            return baseline, "Confirmation pair evidence is inconsistent; using the baseline."
    return candidate, "Using the candidate favored by every recorded confirmation pair; this request may perform differently."


def generate_cpu(
    prompt: str, *, model: str = "qwen2.5-coder:7b", threads: int | None = None,
    benchmark: dict | None = None, tokens: int = 128, context: int = 2048,
    base_url: str = "http://127.0.0.1:11434", timeout_seconds: int = 300,
) -> tuple[str, dict]:
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("prompt must be a nonempty string")
    if type(tokens) is not int or not 1 <= tokens <= 4096:
        raise ValueError("tokens must be an integer in [1, 4096]")
    # Reuse the benchmark's local-address, model, context and timeout guards.
    validate_config({"model": model, "base_url": base_url, "context": context, "timeout_seconds": timeout_seconds})
    client = OllamaClient(base_url, timeout_seconds)
    matches = [m for m in client.request("/api/tags").get("models", []) if m.get("name") == model]
    if len(matches) != 1 or not matches[0].get("digest"):
        raise RuntimeError("Model is not installed with a known digest. This command never downloads models.")
    digest = matches[0]["digest"]
    selected, reason = select_threads(benchmark, model, digest, context, threads)
    payload = {
        "model": model, "prompt": prompt, "keep_alive": "2m",
        "options": {"num_gpu": 0, "num_thread": selected, "num_ctx": context,
                    "num_predict": tokens, "temperature": 0, "seed": 42},
    }
    response, wall, ttft, answer, thinking = client.generate(payload)
    loaded = cpu_status(client.request("/api/ps"), model, digest)
    audit = {
        "model": model, "model_digest": digest, "requested_threads": selected,
        "thread_selection": reason, "context": context, "requested_tokens": tokens,
        "cpu_evidence": {"requested_gpu_layers": 0, "reported_vram_bytes": loaded["size_vram"]},
        "timing": response_metrics(response, wall, ttft),
        "keep_alive": "2m", "thinking_present": bool(thinking),
        "note": "GPU-offload is disabled and Ollama reports zero model VRAM; no hardware-level GPU tracing is performed.",
    }
    return answer, audit


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model", default="qwen2.5-coder:7b")
    parser.add_argument("--threads", type=int, help="Default: 12; an explicit value overrides benchmark selection.")
    parser.add_argument("--benchmark", type=Path, help="Optional completed inference report to select confirmed settings.")
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    args = parser.parse_args()
    try:
        benchmark = json.loads(args.benchmark.read_text(encoding="utf-8-sig")) if args.benchmark else None
        answer, audit = generate_cpu(args.prompt, model=args.model, threads=args.threads, benchmark=benchmark,
                                     tokens=args.tokens, context=args.context, base_url=args.base_url)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.exit(1, f"CPU generation failed: {exc}\n")
    print(answer)
    print(json.dumps(audit, ensure_ascii=False, allow_nan=False, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
