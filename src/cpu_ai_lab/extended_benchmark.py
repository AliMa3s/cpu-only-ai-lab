"""Longer CPU deployment comparisons with independently tuned PyTorch threads.

Raw text completion is a throughput workload, not an instruction/quality test.
Both backends stop naturally. Only exact-budget, exact-prompt-count pairs enter
fixed-work speed summaries, and every incomplete/unequal pair remains visible.
"""
from __future__ import annotations

import argparse
import datetime as dt
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import threading
import time

from .backend_benchmark import VARIANTS, load_json, reset_ollama_runner, sha256, verify_provenance
from .inference import OllamaClient, atomic_write, cpu_status, response_metrics, validate_config


def validate_extended_config(value: dict) -> dict:
    cfg = dict(value)
    validate_config({"base_url": cfg["base_url"], "context": cfg["context"]})
    for key in ("seed", "ollama_threads", "tuning_tokens", "tuning_repeats", "confirmation_repeats"):
        if type(cfg.get(key)) is not int or cfg[key] < (0 if key == "seed" else 1):
            raise ValueError(f"Invalid positive integer setting: {key}")
    for key in ("pytorch_thread_candidates", "token_budgets"):
        entries = cfg.get(key)
        if not isinstance(entries, list) or not entries or len(entries) != len(set(entries)):
            raise ValueError(f"Expected distinct values in {key}")
        if any(type(x) is not int or x < 1 for x in entries):
            raise ValueError(f"Expected positive integers in {key}")
    if max(cfg["token_budgets"] + [cfg["tuning_tokens"]]) >= cfg["context"]:
        raise ValueError("Output budget must fit the configured context")
    seen_ids, seen_text = set(), set()
    for split in ("tuning_prompts", "confirmation_prompts"):
        prompts = cfg.get(split)
        if not isinstance(prompts, list) or not prompts:
            raise ValueError(f"Expected nonempty {split}")
        for prompt in prompts:
            if not isinstance(prompt, dict) or any(not isinstance(prompt.get(k), str) or not prompt[k].strip() for k in ("id", "text")):
                raise ValueError("Every prompt needs a nonempty id and text")
            if prompt["id"] in seen_ids or prompt["text"] in seen_text:
                raise ValueError("Tuning and confirmation prompts must be distinct, including IDs")
            seen_ids.add(prompt["id"])
            seen_text.add(prompt["text"])
    return cfg


def make_schedule(prompts: list[dict], budgets: list[int], repeats: int, seed: int) -> list[dict]:
    schedule = [{"pair_id": f"{p['id']}-{budget}-r{repeat}", "prompt_id": p["id"],
                 "requested_tokens": budget, "repeat": repeat}
                for repeat in range(1, repeats + 1) for p in prompts for budget in budgets]
    rng = random.Random(seed)
    rng.shuffle(schedule)
    offset = rng.randrange(2)
    for index, row in enumerate(schedule):
        row["order"] = list(VARIANTS if (index + offset) % 2 == 0 else reversed(VARIANTS))
    return schedule


def select_threads(records: list[dict], candidates: list[int], expected_groups: int) -> dict:
    """Choose total latency only after checking equal work in every tuning group."""
    groups = {}
    for row in records:
        group = groups.setdefault((row["prompt_id"], row["repeat"]), {})
        if row["threads"] in group:
            raise ValueError("Duplicate thread count in a tuning group")
        group[row["threads"]] = row
    if len(groups) != expected_groups:
        raise ValueError("Incomplete thread tuning groups")
    for group in groups.values():
        if set(group) != set(candidates):
            raise ValueError("Incomplete thread candidates")
        if len({(r["output_tokens"], r["output"]) for r in group.values()}) != 1:
            raise ValueError("Thread tuning changed generated work; refusing a latency-based choice")
    totals = {str(threads): sum(g[threads]["wall_seconds"] for g in groups.values()) for threads in candidates}
    selected = min(candidates, key=lambda threads: (totals[str(threads)], threads))
    return {"selected_threads": selected, "total_wall_seconds_by_threads": totals,
            "criterion": "Lowest summed wall time on separate tuning prompts; identical outputs and lengths required across candidates",
            "tuning_groups": len(groups), "confirmation_used_for_selection": False}


def summarize(records: list[dict]) -> dict:
    grouped = {}
    for row in records:
        if row["variant"] not in VARIANTS:
            raise ValueError("Unknown variant")
        pair = grouped.setdefault(row["pair_id"], {})
        if row["variant"] in pair:
            raise ValueError("Duplicate variant in a timing pair")
        pair[row["variant"]] = row
    included, other = [], []
    for pair_id, pair in grouped.items():
        if set(pair) != set(VARIANTS):
            other.append({"pair_id": pair_id, "reasons": ["incomplete pair"]})
            continue
        baseline, candidate = pair[VARIANTS[0]], pair[VARIANTS[1]]
        budget = baseline["requested_tokens"]
        if candidate["requested_tokens"] != budget:
            raise ValueError("Different requested token budgets in a pair")
        for row in pair.values():
            if not isinstance(row["wall_seconds"], (float, int)) or not math.isfinite(row["wall_seconds"]) or row["wall_seconds"] <= 0:
                raise ValueError("Invalid timing")
        reasons = []
        if baseline["output_tokens"] != budget or candidate["output_tokens"] != budget:
            reasons.append("At least one backend stopped before the requested token budget")
        if baseline["prompt_tokens"] != candidate["prompt_tokens"]:
            reasons.append("Prompt token counts differ")
        if candidate.get("prompt_cached_tokens", 0) not in (0, None):
            reasons.append("Ollama reported cached prompt tokens")
        detail = {"pair_id": pair_id, "prompt_id": baseline["prompt_id"], "requested_tokens": budget,
                  "pytorch_tokens": baseline["output_tokens"], "ollama_tokens": candidate["output_tokens"],
                  "pytorch_seconds": baseline["wall_seconds"], "ollama_seconds": candidate["wall_seconds"],
                  "outputs_identical": baseline["output"] == candidate["output"],
                  "pytorch_prompt_tokens": baseline["prompt_tokens"], "ollama_prompt_tokens": candidate["prompt_tokens"]}
        if reasons:
            other.append({**detail, "reasons": reasons})
        else:
            included.append({**detail, "paired_speedup": baseline["wall_seconds"] / candidate["wall_seconds"]})

    def aggregate(pairs):
        if not pairs:
            return {"pairs": 0}
        ratios = [p["paired_speedup"] for p in pairs]
        tokens = sum(p["requested_tokens"] for p in pairs)
        return {"pairs": len(pairs), "median_paired_speedup": statistics.median(ratios),
                "min_paired_speedup": min(ratios), "max_paired_speedup": max(ratios),
                "ollama_faster_pairs": sum(r > 1 for r in ratios),
                "pytorch_aggregate_tokens_per_second": tokens / sum(p["pytorch_seconds"] for p in pairs),
                "ollama_aggregate_tokens_per_second": tokens / sum(p["ollama_seconds"] for p in pairs),
                "identical_output_pairs": sum(p["outputs_identical"] for p in pairs)}

    per_variant = {}
    for variant in VARIANTS:
        rows = [r for r in records if r["variant"] == variant]
        first = [r["first_content_seconds"] for r in rows if r.get("first_content_seconds") is not None]
        per_variant[variant] = {
            "requests": len(rows), "natural_early_stops": sum(r["output_tokens"] < r["requested_tokens"] for r in rows),
            "median_first_content_seconds": statistics.median(first) if first else None,
            "ttft_observed_requests": len(first),
            "sampled_peak_python_rss_bytes": max((r.get("memory", {}).get("python_peak_rss_bytes", 0) for r in rows), default=0),
            "sampled_peak_ollama_process_rss_sum_bytes": max((r.get("memory", {}).get("ollama_peak_rss_sum_bytes", 0) for r in rows), default=0),
        }
    return {"status": "measured" if included else "no_exact_budget_pairs", "scheduled_pairs_observed": len(grouped),
            "fixed_work": aggregate(included), "by_token_budget": {str(b): aggregate([p for p in included if p["requested_tokens"] == b]) for b in sorted({r["requested_tokens"] for r in records})},
            "by_prompt": {p: aggregate([r for r in included if r["prompt_id"] == p]) for p in sorted({r["prompt_id"] for r in records})},
            "included_pairs": included, "unmatched_or_early_stop_pairs": other, "all_requests": per_variant,
            "interpretation": "Descriptive paired throughput; exact token counts do not imply identical token sequences or useful long-form answers. Every request is retained."}


def ollama_process_tree(snapshot: list[dict]) -> list[dict]:
    """Select Ollama roots and all descendants, including native llama-server.

    The input is a fresh process snapshot at each sampling instant. Only the
    four public measurement fields below are retained; no command lines, paths,
    usernames, environment values, or unrelated processes are recorded.
    """
    by_pid = {row["pid"]: row for row in snapshot}
    selected = {pid for pid, row in by_pid.items() if (row.get("name") or "").casefold().startswith("ollama")}
    while True:
        children = {pid for pid, row in by_pid.items() if row.get("ppid") in selected}
        expanded = selected | children
        if expanded == selected:
            break
        selected = expanded
    return [{key: by_pid[pid].get(key) for key in ("pid", "ppid", "name", "rss_bytes")} for pid in sorted(selected)]


class MemorySampler:
    """Sample RSS using a freshly discovered process tree on every sample."""
    def __init__(self):
        self.stop_event = threading.Event()
        self.result = {"python_peak_rss_bytes": 0, "ollama_peak_rss_sum_bytes": 0, "samples": 0,
                       "ollama_process_names": [], "sample_interval_seconds": 0.1,
                       "method": "ollama-roots-and-recursive-descendants-refreshed-each-sample-v2",
                       "ollama_processes": [], "ollama_peak_process_snapshot": [],
                       "unreadable_ollama_rss_observations": 0, "process_scan_failures": 0,
                       "python_rss_failures": 0,
                       "note": "RSS samples, not exact peaks. Every sample rediscovers Ollama roots and their descendants regardless of executable name, including late-started runners. RSS sums can double-count shared pages; Python RSS is separate. Only PID, parent PID, executable name, and RSS evidence is retained for the Ollama tree."}
        self.observed = {}

    def __enter__(self):
        import psutil
        self.process = psutil.Process()
        self._sample_once()
        self.thread = threading.Thread(target=self._sample, daemon=True)
        self.thread.start()
        return self

    def _read_process_snapshot(self):
        import psutil
        snapshot = []
        for process in psutil.process_iter(["pid", "ppid", "name", "memory_info"], ad_value=None):
            try:
                info = process.info
                memory = info.get("memory_info")
                snapshot.append({"pid": info["pid"], "ppid": info.get("ppid"), "name": info.get("name"),
                                 "rss_bytes": memory.rss if memory is not None else None})
            except psutil.Error:
                continue  # A process may exit between discovery and collection.
        return snapshot

    def _sample_once(self):
        import psutil
        try:
            self.result["python_peak_rss_bytes"] = max(self.result["python_peak_rss_bytes"], self.process.memory_info().rss)
        except psutil.Error:
            self.result["python_rss_failures"] += 1
        try:
            processes = ollama_process_tree(self._read_process_snapshot())
        except (psutil.Error, OSError):
            self.result["process_scan_failures"] += 1
            processes = []
        total = sum(row["rss_bytes"] for row in processes if isinstance(row["rss_bytes"], int))
        if total > self.result["ollama_peak_rss_sum_bytes"]:
            self.result["ollama_peak_rss_sum_bytes"] = total
            self.result["ollama_peak_process_snapshot"] = processes
        for row in processes:
            if not isinstance(row["rss_bytes"], int):
                self.result["unreadable_ollama_rss_observations"] += 1
            previous = self.observed.get(row["pid"])
            if previous is None or (row["rss_bytes"] or 0) > (previous["rss_bytes"] or 0):
                self.observed[row["pid"]] = dict(row)
        self.result["ollama_processes"] = [self.observed[pid] for pid in sorted(self.observed)]
        self.result["ollama_process_names"] = sorted({r["name"] for r in self.observed.values() if r["name"]})
        self.result["samples"] += 1

    def _sample(self):
        while not self.stop_event.wait(0.1):
            self._sample_once()

    def __exit__(self, *_):
        self.stop_event.set()
        self.thread.join()
        self._sample_once()


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError("Choose a new output path; existing measurements are preserved")
    cfg = validate_extended_config(load_json(args.config))
    source_cfg, training_metrics, manifest = verify_provenance(args.training_dir, args.deployment_manifest)
    client = OllamaClient(cfg["base_url"], 600)
    if client.request("/api/ps").get("models"):
        raise RuntimeError("Run this experiment while Ollama has no models loaded")
    model_name, digest = manifest["ollama_model"], manifest["ollama_digest"]
    tags = [m for m in client.request("/api/tags").get("models", []) if m.get("name") == model_name]
    if len(tags) != 1 or tags[0].get("digest") != digest or tags[0].get("details", {}).get("quantization_level") != "F16":
        raise ValueError("The installed F16 model does not match the verified deployment")
    report = {
        "format": "cpu-ai-lab-extended-benchmark-v1", "status": "running",
        "started_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "configuration": cfg,
        "provenance": {"model_id": source_cfg["model_id"], "revision": source_cfg["revision"],
                       "adapter_sha256": training_metrics["artifact_sha256"], "ollama_digest": digest,
                       "deployment_manifest_sha256": sha256(args.deployment_manifest), "config_sha256": sha256(args.config),
                       "gguf_sha256": manifest.get("gguf_sha256")},
        "ollama_version": client.request("/api/version"), "ollama_model": model_name,
        "method": {
            "workload": "Raw text continuation with natural EOS on both backends, no chat template. Predetermined prompts and 128/256-token caps; every run retained.",
            "baseline": "Merged trained FP32 PyTorch, eval mode, SDPA, KV cache; threads selected only on separate tuning prompts.",
            "candidate": "The same merged checkpoint exported to F16 GGUF in Ollama/llama.cpp; runtime and storage precision both change.",
            "wall": "Tokenization + generation + response decoding. Ollama also includes local HTTP/stream parsing. Model loading and explicit cache resets are excluded.",
            "ttft": "First nonempty decoded content at the local client. PyTorch additionally records its first generated token callback. Ollama chunks may contain more than one token.",
            "prefill_decode": "PyTorch forward-hook prefill/decode compute durations and Ollama API prompt/eval durations are retained with distinct names; they are not identical timing boundaries.",
            "cache": "Fresh PyTorch generation cache. Ollama explicitly unloaded and reloaded without a prompt for every measured request; first-call kernel overhead is included.",
            "fixed_work_policy": "Require both outputs to reach the exact requested token budget, identical prompt token counts, and no reported cached prompt tokens. Retain and enumerate all other pairs separately.",
            "quality": "Throughput prompts are not scored for usefulness. Prior extraction quality results do not establish long-form quality; raw outputs are saved for inspection.",
            "limitations": "One machine, four prompt families, short repetitions, no energy telemetry or hardware-level GPU tracing. Memory RSS includes shared pages. No broad quality-equivalence claim.",
        },
        "warmups": [], "tuning": [], "timing": [],
    }
    atomic_write(args.output, report)
    started_ollama = False
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .training import assert_cpu_model
        if torch.version.cuda is not None or torch.cuda.is_available():
            raise RuntimeError("A CPU-only PyTorch build is required")
        torch.set_num_threads(cfg["pytorch_thread_candidates"][0])
        torch.set_num_interop_threads(1)
        torch.manual_seed(cfg["seed"])
        torch.use_deterministic_algorithms(True)
        options = {"revision": source_cfg["revision"], "local_files_only": True, "trust_remote_code": False}
        tokenizer = AutoTokenizer.from_pretrained(source_cfg["model_id"], **options)
        tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token_id is None else tokenizer.pad_token
        base = AutoModelForCausalLM.from_pretrained(source_cfg["model_id"], torch_dtype=torch.float32, attn_implementation="sdpa", **options).cpu()
        model = PeftModel.from_pretrained(base, str(args.training_dir / "adapter"), is_trainable=False).merge_and_unload(safe_merge=True).cpu().float().eval()
        model.config.use_cache = True
        model.generation_config.do_sample = False
        for name in ("temperature", "top_p", "top_k"):
            setattr(model.generation_config, name, None)
        assert_cpu_model(model)
        report["versions"] = {"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft", "psutil")}}
        report["cpu_evidence"] = {"torch_cuda_build": torch.version.cuda, "torch_cuda_available": torch.cuda.is_available(),
                                  "parameter_devices": sorted({str(p.device) for p in model.parameters()}),
                                  "buffer_devices": sorted({str(p.device) for p in model.buffers()}), "interop_threads": torch.get_num_interop_threads()}
        all_prompts = cfg["tuning_prompts"] + cfg["confirmation_prompts"]
        prompt_counts = {p["id"]: len(tokenizer.encode(p["text"], add_special_tokens=False)) for p in all_prompts}
        if any(n + max(cfg["token_budgets"] + [cfg["tuning_tokens"]]) > cfg["context"] for n in prompt_counts.values()):
            raise ValueError("A full prompt plus output budget exceeds context; no truncation allowed")
        report["hf_prompt_token_counts"] = prompt_counts

        def pytorch_one(prompt: dict, budget: int, threads: int) -> dict:
            torch.set_num_threads(threads)
            events, forwards, forward_start = [], [], [0.0]
            class TokenTimer:
                skip_prompt = True
                first_content = None
                pending_ids = []
                def put(self, value):
                    if self.skip_prompt:
                        self.skip_prompt = False
                        return
                    events.append(time.perf_counter())
                    if self.first_content is None:
                        self.pending_ids.extend(value.reshape(-1).tolist())
                        if tokenizer.decode(self.pending_ids, skip_special_tokens=True):
                            self.first_content = time.perf_counter()
                def end(self):
                    pass
            timer = TokenTimer()
            def before_forward(*_):
                forward_start[0] = time.perf_counter()
            def after_forward(*_):
                forwards.append(time.perf_counter() - forward_start[0])
            pre_hook = model.register_forward_pre_hook(before_forward)
            post_hook = model.register_forward_hook(after_forward)
            try:
                with MemorySampler() as memory, torch.inference_mode():
                    started = time.perf_counter()
                    inputs = tokenizer(prompt["text"], add_special_tokens=False, return_tensors="pt")
                    tokenization_seconds = time.perf_counter() - started
                    generated = model.generate(**inputs, max_new_tokens=budget, min_new_tokens=0,
                                               do_sample=False, use_cache=True, repetition_penalty=1.0,
                                               pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id,
                                               streamer=timer)
                    new_ids = generated[0, inputs["input_ids"].shape[1]:]
                    output = tokenizer.decode(new_ids, skip_special_tokens=True)
                    elapsed = time.perf_counter() - started
            finally:
                pre_hook.remove()
                post_hook.remove()
            after_first = events[-1] - events[0] if len(events) > 1 else None
            return {"variant": "pytorch_fp32", "threads": threads, "prompt_id": prompt["id"],
                    "requested_tokens": budget, "prompt_tokens": int(inputs["input_ids"].numel()),
                    "output_tokens": int(new_ids.numel()), "output": output, "wall_seconds": elapsed,
                    "first_content_seconds": timer.first_content - started if timer.first_content is not None else None,
                    "first_generated_token_seconds": events[0] - started if events else None,
                    "tokenization_seconds": tokenization_seconds,
                    "prefill_forward_seconds": forwards[0] if forwards else None,
                    "decode_forward_seconds": sum(forwards[1:]), "model_forward_calls": len(forwards),
                    "after_first_token_seconds": after_first,
                    "after_first_token_tokens_per_second": (len(events) - 1) / after_first if after_first else None,
                    "ended_with_eos": bool(len(new_ids) and int(new_ids[-1]) == tokenizer.eos_token_id),
                    "memory": memory.result}

        def ollama_one(prompt: dict, budget: int) -> dict:
            nonlocal started_ollama
            started_ollama = True
            reset = reset_ollama_runner(client, model_name, digest, cfg["ollama_threads"], cfg["context"])
            payload = {"model": model_name, "prompt": prompt["text"], "raw": True, "keep_alive": "10m",
                       "options": {"num_gpu": 0, "num_thread": cfg["ollama_threads"], "num_ctx": cfg["context"],
                                   "num_predict": budget, "temperature": 0, "seed": cfg["seed"], "repeat_penalty": 1.0,
                                   "stop": []}}
            with MemorySampler() as memory:
                response, elapsed, ttft, output, thinking = client.generate(payload)
            status = cpu_status(client.request("/api/ps"), model_name, digest)
            measured = response_metrics(response, elapsed, ttft)
            return {"variant": "ollama_f16", "threads": cfg["ollama_threads"], "prompt_id": prompt["id"],
                    "requested_tokens": budget, "prompt_tokens": response.get("prompt_eval_count"),
                    "hf_prompt_tokens": prompt_counts[prompt["id"]], "prompt_cached_tokens": response.get("prompt_eval_cached_count"),
                    "output_tokens": measured["output_tokens"], "output": output, "thinking": thinking,
                    "wall_seconds": elapsed, "first_content_seconds": ttft,
                    "ollama_server_prefill_seconds": measured["prompt_seconds"],
                    "ollama_server_decode_seconds": measured["decode_seconds"],
                    "ollama_server_decode_tokens_per_second": measured["decode_tokens_per_second"],
                    "raw_timing_record": response, "options": payload["options"], "raw": True,
                    "memory": memory.result, "cache_reset": reset,
                    "cpu_evidence": {"size_vram": status["size_vram"], "digest": status["digest"]}}

        for threads in cfg["pytorch_thread_candidates"]:
            report["warmups"].append(pytorch_one(cfg["tuning_prompts"][0], 8, threads))
        tuning_schedule = [(p, repeat, t) for repeat in range(1, cfg["tuning_repeats"] + 1)
                           for p in cfg["tuning_prompts"] for t in cfg["pytorch_thread_candidates"]]
        random.Random(cfg["seed"]).shuffle(tuning_schedule)
        for prompt, repeat, threads in tuning_schedule:
            row = {**pytorch_one(prompt, cfg["tuning_tokens"], threads), "repeat": repeat}
            report["tuning"].append(row)
            atomic_write(args.output, report)
            print(json.dumps({"stage": "tuning", "prompt": prompt["id"], "threads": threads,
                              "tokens": row["output_tokens"], "wall_seconds": row["wall_seconds"]}), flush=True)
        selection = select_threads(report["tuning"], cfg["pytorch_thread_candidates"], len(cfg["tuning_prompts"]) * cfg["tuning_repeats"])
        report["thread_selection"] = selection
        schedule = make_schedule(cfg["confirmation_prompts"], cfg["token_budgets"], cfg["confirmation_repeats"], cfg["seed"] + 1)
        report["confirmation_schedule"] = schedule
        prompts = {p["id"]: p for p in cfg["confirmation_prompts"]}
        for pair in schedule:
            for variant in pair["order"]:
                prompt, budget = prompts[pair["prompt_id"]], pair["requested_tokens"]
                row = pytorch_one(prompt, budget, selection["selected_threads"]) if variant == "pytorch_fp32" else ollama_one(prompt, budget)
                row.update({"pair_id": pair["pair_id"], "repeat": pair["repeat"]})
                report["timing"].append(row)
                atomic_write(args.output, report)
                print(json.dumps({"stage": "confirmation", "pair": pair["pair_id"], "variant": variant,
                                  "tokens": row["output_tokens"], "requested": budget, "wall_tps": row["output_tokens"] / row["wall_seconds"]}), flush=True)
        report["summary"] = summarize(report["timing"])
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if started_ollama:
            try:
                client.request("/api/generate", {"model": model_name, "keep_alive": 0, "stream": False, "options": {"num_gpu": 0}})
                report["cleanup"] = "Test model unloaded" if not client.request("/api/ps").get("models") else "Other models remain loaded"
            except Exception as exc:
                report["cleanup"] = f"Unload failed: {exc}"
        report["finished_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_write(args.output, report)
    print(json.dumps({"status": report["status"], "selected_threads": report["thread_selection"]["selected_threads"],
                      "fixed_work": report["summary"]["fixed_work"],
                      "unmatched_pairs": len(report["summary"]["unmatched_or_early_stop_pairs"])}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/extended-benchmark.json"))
    parser.add_argument("--training-dir", type=Path, default=Path("artifacts/train-run"))
    parser.add_argument("--deployment-manifest", type=Path, default=Path("artifacts/deployment-f16/deployment-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/extended-benchmark.json"))
    run(parser.parse_args())


if __name__ == "__main__":
    main()
