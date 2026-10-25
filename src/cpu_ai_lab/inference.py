"""CPU-only Ollama thread tuning with a separate held-out confirmation phase.

Uses only the Python standard library. The target model must already be installed.
No training, quantization conversion, downloads or GPU calls are performed here.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import time
import urllib.parse
import urllib.request


DEFAULT_CONFIG = {
    "model": "qwen2.5-coder:7b",
    "base_url": "http://127.0.0.1:11434",
    "threads": [2, 4, 6, 8, 12],
    "baseline_threads": 12,
    "context": 2048,
    "tokens": 64,
    "warmup_tokens": 8,
    "confirmation_repeats": 2,
    "seed": 20261025,
    "timeout_seconds": 300,
    "tuning_prompts": [
        "Explain how merge sort works, including its time complexity, then write a Python implementation.",
        "Write a Python function that counts word frequencies in a text file, then explain its edge cases.",
        "Explain the difference between a SQL inner join and a left join, with a concrete example of each.",
    ],
    "confirmation_prompts": [
        "Write a Python breadth-first search for the shortest path in an unweighted graph and explain its complexity.",
        "Explain why database transactions need atomicity and isolation, with practical examples.",
        "Write a Python function to merge overlapping intervals and explain how you would test it.",
    ],
}


def validate_config(config: dict) -> dict:
    cfg = {**DEFAULT_CONFIG, **config}
    if not isinstance(cfg["model"], str) or not cfg["model"].strip():
        raise ValueError("model must be a nonempty installed model name")
    parsed = urllib.parse.urlparse(cfg["base_url"])
    if parsed.scheme != "http" or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("base_url must point to a local HTTP Ollama server")
    if parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("base_url must contain only the local server address and port")
    for key, lower, upper in (
        ("baseline_threads", 1, 256), ("context", 256, 32768),
        ("tokens", 16, 4096), ("warmup_tokens", 1, 128),
        ("confirmation_repeats", 1, 20), ("seed", 0, 2**31 - 1),
        ("timeout_seconds", 5, 3600),
    ):
        if type(cfg[key]) is not int or not lower <= cfg[key] <= upper:
            raise ValueError(f"{key} must be an integer in [{lower}, {upper}]")
    threads = cfg["threads"]
    if not isinstance(threads, list) or not threads or len(set(threads)) != len(threads):
        raise ValueError("threads must be a nonempty list of distinct integers")
    if any(type(t) is not int or not 1 <= t <= 256 for t in threads):
        raise ValueError("thread counts must be integers in [1, 256]")
    if cfg["baseline_threads"] not in threads:
        raise ValueError("baseline_threads must appear in threads")
    for key in ("tuning_prompts", "confirmation_prompts"):
        prompts = cfg[key]
        if not isinstance(prompts, list) or not prompts or any(not isinstance(p, str) or not p.strip() for p in prompts):
            raise ValueError(f"{key} must be a nonempty list of nonempty strings")
        if len(set(prompts)) != len(prompts):
            raise ValueError(f"{key} contains duplicate prompts")
    if set(cfg["tuning_prompts"]) & set(cfg["confirmation_prompts"]):
        raise ValueError("confirmation prompts must be distinct from tuning prompts")
    return cfg


def cpu_status(status: dict, model: str, expected_digest: str | None = None) -> dict:
    matches = [entry for entry in status.get("models", []) if entry.get("name") == model]
    if len(matches) != 1:
        raise RuntimeError("Expected exactly one running entry for the tested model")
    loaded = matches[0]
    if type(loaded.get("size_vram")) is not int or loaded["size_vram"] != 0:
        raise RuntimeError("CPU-only verification failed: model VRAM allocation is not confirmed zero")
    if expected_digest and loaded.get("digest") != expected_digest:
        raise RuntimeError("The model digest changed during the experiment")
    return loaded


def response_metrics(response: dict, wall_seconds: float, ttft_seconds: float | None) -> dict:
    count, duration = response.get("eval_count"), response.get("eval_duration")
    if type(count) is not int or count <= 0 or not isinstance(duration, (int, float)) or duration <= 0:
        raise RuntimeError("Missing or invalid generated-token timing")
    if not math.isfinite(float(duration)) or not math.isfinite(wall_seconds) or wall_seconds <= 0:
        raise RuntimeError("Nonfinite timing")
    return {
        "output_tokens": count,
        "decode_seconds": duration / 1e9,
        "decode_tokens_per_second": count * 1e9 / duration,
        "request_wall_seconds": wall_seconds,
        "first_content_token_seconds": ttft_seconds,
        "server_total_seconds": response.get("total_duration", 0) / 1e9,
        "load_seconds": response.get("load_duration", 0) / 1e9,
        "prompt_tokens_evaluated": response.get("prompt_eval_count"),
        "prompt_cached_tokens": response.get("prompt_eval_cached_count"),
        "prompt_seconds": response.get("prompt_eval_duration", 0) / 1e9,
        "done_reason": response.get("done_reason"),
    }


def summarize(records: list[dict]) -> dict:
    grouped: dict[int, list[dict]] = {}
    for row in records:
        grouped.setdefault(row["threads"], []).append(row)
    result = {}
    for threads, rows in sorted(grouped.items()):
        rates = [r["decode_tokens_per_second"] for r in rows]
        total_tokens = sum(r["output_tokens"] for r in rows)
        total_seconds = sum(r["decode_seconds"] for r in rows)
        result[str(threads)] = {
            "samples": len(rows),
            "median_decode_tokens_per_second": statistics.median(rates),
            "aggregate_decode_tokens_per_second": total_tokens / total_seconds,
            "min_decode_tokens_per_second": min(rates),
            "max_decode_tokens_per_second": max(rates),
            "median_request_wall_seconds": statistics.median(r["request_wall_seconds"] for r in rows),
            "median_first_content_token_seconds": statistics.median(r["first_content_token_seconds"] for r in rows if r["first_content_token_seconds"] is not None)
            if any(r["first_content_token_seconds"] is not None for r in rows) else None,
            "output_token_counts": [r["output_tokens"] for r in rows],
        }
    return result


def paired_summary(records: list[dict], baseline: int, candidate: int) -> dict:
    if baseline == candidate:
        return {"status": "baseline_selected", "speedup": 1.0, "improvement_percent": 0.0,
                "note": "Tuning did not select a different thread count; no improvement claimed."}
    pairs: dict[str, dict[int, dict]] = {}
    for row in records:
        pairs.setdefault(row["pair_id"], {})[row["threads"]] = row
    ratios, details = [], []
    for pair_id, rows in sorted(pairs.items()):
        if baseline not in rows or candidate not in rows:
            continue
        b, c = rows[baseline], rows[candidate]
        ratio = c["decode_tokens_per_second"] / b["decode_tokens_per_second"]
        ratios.append(ratio)
        details.append({"pair_id": pair_id, "speedup": ratio,
                        "baseline_tokens": b["output_tokens"], "candidate_tokens": c["output_tokens"],
                        "outputs_identical": b["output"] == c["output"]})
    if not ratios:
        return {"status": "no_complete_pairs", "pairs": []}
    median = statistics.median(ratios)
    return {
        "status": "measured", "baseline_threads": baseline, "candidate_threads": candidate,
        "complete_pairs": len(ratios), "median_paired_speedup": median,
        "improvement_percent": (median - 1) * 100,
        "geometric_mean_paired_speedup": math.exp(statistics.mean(math.log(x) for x in ratios)),
        "candidate_faster_pairs": sum(x > 1 for x in ratios),
        "all_output_lengths_equal": all(p["baseline_tokens"] == p["candidate_tokens"] for p in details),
        "pairs": details,
        "note": "Descriptive paired measurement on this machine; no statistical significance or quality equivalence claim.",
    }


class OllamaClient:
    def __init__(self, base_url: str, timeout: int):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout

    def request(self, path: str, payload: dict | None = None) -> dict:
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(self.base_url + path, data=data, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            result = json.load(response)
        if "error" in result:
            raise RuntimeError(str(result["error"]))
        return result

    def generate(self, payload: dict) -> tuple[dict, float, float | None, str, str]:
        request = urllib.request.Request(
            self.base_url + "/api/generate", data=json.dumps({**payload, "stream": True}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        start = time.perf_counter()
        first_content, final = None, None
        output, thinking = [], []
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            for line in response:
                if not line.strip():
                    continue
                chunk = json.loads(line)
                if "error" in chunk:
                    raise RuntimeError(str(chunk["error"]))
                if chunk.get("response"):
                    if first_content is None:
                        first_content = time.perf_counter() - start
                    output.append(chunk["response"])
                if chunk.get("thinking"):
                    thinking.append(chunk["thinking"])
                if chunk.get("done"):
                    final = chunk
        elapsed = time.perf_counter() - start
        if final is None:
            raise RuntimeError("Generation stream ended without a final timing record")
        return final, elapsed, first_content, "".join(output), "".join(thinking)


def atomic_write(path: Path, report: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def run_experiment(cfg: dict, output: Path, phase: str = "all", candidate_threads: int | None = None) -> dict:
    cfg = validate_config(cfg)
    if phase not in {"all", "tuning", "confirmation"}:
        raise ValueError("Unknown experiment phase")
    if phase == "confirmation" and (type(candidate_threads) is not int or candidate_threads not in cfg["threads"]):
        raise ValueError("Confirmation requires --candidate-threads from the configured thread counts")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite an existing result: {output}")
    client = OllamaClient(cfg["base_url"], cfg["timeout_seconds"])
    if client.request("/api/ps").get("models"):
        raise RuntimeError("Ollama already has a model loaded. Run this experiment when it is idle.")
    installed = client.request("/api/tags").get("models", [])
    match = [m for m in installed if m.get("name") == cfg["model"]]
    if len(match) != 1:
        raise RuntimeError("The requested model must already be installed; no automatic download is attempted")
    model_info = match[0]
    digest = model_info.get("digest")
    if not digest:
        raise RuntimeError("Installed model metadata does not supply an immutable digest")
    report = {
        "schema_version": 1, "status": "running", "phase": phase,
        "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "config": cfg, "model": model_info,
        "ollama_version": client.request("/api/version"),
        "hardware": {"platform": platform.platform(), "cpu": platform.processor(), "logical_cpus": os.cpu_count()},
        "method": {
            "primary_metric": "Generated tokens per second from eval_count/eval_duration; excludes prefill and loading.",
            "selection": "Highest median decoding throughput on tuning prompts; held-out confirmation reports separate paired results.",
            "cpu_verification": "Every request sets num_gpu=0; every response must have matching model digest and size_vram=0 in /api/ps.",
            "cache": "Model is warm before measurements. Prompt caching and filesystem cache are not controlled; available cached-token counts are recorded.",
            "timing": "First content-token latency uses streamed response arrival, not the server prompt timer.",
            "limits": "No hardware-level GPU trace, energy measurement, semantic quality evaluation, or control of other system load. Short outputs and prompts only.",
        },
        "warmups": [], "tuning": [], "confirmation": [],
    }
    atomic_write(output, report)
    rng = random.Random(cfg["seed"])
    last_threads = None
    started_model = False

    def run_one(threads: int, prompt: str, tokens: int, group: str, extra: dict) -> dict:
        nonlocal started_model
        payload = {
            "model": cfg["model"], "prompt": prompt, "keep_alive": "10m",
            "options": {"num_gpu": 0, "num_thread": threads, "num_ctx": cfg["context"],
                        "num_predict": tokens, "temperature": 0, "seed": cfg["seed"]},
        }
        started_model = True
        response, wall, ttft, text, thinking = client.generate(payload)
        status = cpu_status(client.request("/api/ps"), cfg["model"], digest)
        row = {
            "sequence": sum(len(report[k]) for k in ("warmups", "tuning", "confirmation")) + 1,
            "threads": threads, "prompt": prompt, "requested_tokens": tokens,
            "options": payload["options"], **response_metrics(response, wall, ttft),
            "output": text, "thinking": thinking, "raw_timing_record": response,
            "cpu_evidence": {"size_vram": status["size_vram"], "model_digest": status["digest"],
                             "reported_model_bytes": status.get("size"), "details": status.get("details")},
            **extra,
        }
        report[group].append(row)
        atomic_write(output, report)
        print(json.dumps({"phase": group, "threads": threads, "tokens": row["output_tokens"],
                          "decode_tps": round(row["decode_tokens_per_second"], 3), **extra}), flush=True)
        return row

    def prepare(threads: int) -> None:
        nonlocal last_threads
        if last_threads != threads:
            run_one(threads, "Explain what a programming variable is.", cfg["warmup_tokens"], "warmups", {})
            last_threads = threads

    try:
        if phase in {"all", "tuning"}:
            order = list(cfg["threads"])
            rng.shuffle(order)
            report["tuning_thread_order"] = order
            for threads in order:
                prepare(threads)
                prompt_order = list(enumerate(cfg["tuning_prompts"]))
                rng.shuffle(prompt_order)
                for index, prompt in prompt_order:
                    run_one(threads, prompt, cfg["tokens"], "tuning", {"prompt_index": index})
            summary = summarize(report["tuning"])
            candidate_threads = max(cfg["threads"], key=lambda t: (summary[str(t)]["median_decode_tokens_per_second"], -t))
            report["tuning_summary"] = summary
            report["selected_threads"] = candidate_threads
            atomic_write(output, report)
        if phase in {"all", "confirmation"}:
            report["selected_threads"] = candidate_threads
            if candidate_threads == cfg["baseline_threads"]:
                report["confirmation_summary"] = paired_summary([], cfg["baseline_threads"], candidate_threads)
            else:
                pairs = [(rep, index, prompt) for rep in range(cfg["confirmation_repeats"])
                         for index, prompt in enumerate(cfg["confirmation_prompts"])]
                rng.shuffle(pairs)
                for repeat, index, prompt in pairs:
                    order = [cfg["baseline_threads"], candidate_threads]
                    rng.shuffle(order)
                    for threads in order:
                        prepare(threads)
                        run_one(threads, prompt, cfg["tokens"], "confirmation", {
                            "pair_id": f"repeat-{repeat + 1}-prompt-{index + 1}", "repeat": repeat + 1, "prompt_index": index,
                        })
                report["confirmation_by_threads"] = summarize(report["confirmation"])
                report["confirmation_summary"] = paired_summary(report["confirmation"], cfg["baseline_threads"], candidate_threads)
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if started_model:
            try:
                current = client.request("/api/ps").get("models", [])
                if any(m.get("name") == cfg["model"] and m.get("digest") == digest for m in current):
                    client.request("/api/generate", {"model": cfg["model"], "keep_alive": 0, "stream": False, "options": {"num_gpu": 0}})
                report["cleanup"] = "Test model unloaded or already absent"
            except Exception as exc:
                report["cleanup"] = f"Unload failed; keep-alive will expire: {type(exc).__name__}: {exc}"
        report["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_write(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/inference.json"))
    parser.add_argument("--phase", choices=("all", "tuning", "confirmation"), default="all")
    parser.add_argument("--candidate-threads", type=int)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text(encoding="utf-8-sig")) if args.config else {}
    report = run_experiment(cfg, args.output, args.phase, args.candidate_threads)
    print(json.dumps({"result": str(args.output), "selected_threads": report.get("selected_threads"),
                      "confirmation": report.get("confirmation_summary")}, indent=2), flush=True)


if __name__ == "__main__":
    main()
