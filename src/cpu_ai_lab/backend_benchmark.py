"""Compare a hash-checked trained adapter on CPU PyTorch and CPU llama.cpp.

The comparison changes both runtime and weight-storage precision (FP32 -> F16).
It is not a pure backend experiment, and does not claim broad quality equivalence.
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import time

from .data import score_response
from .inference import OllamaClient, atomic_write, cpu_status, response_metrics, validate_config


VARIANTS = ("pytorch_fp32", "ollama_f16")
TIMING_PROMPTS = [
    "Write a detailed tutorial explaining merge sort to a beginner. Describe splitting, recursion, merging, runtime complexity, and a worked example. Use several paragraphs before providing Python code.",
    "Write a detailed tutorial explaining database transactions. Discuss atomicity, consistency, isolation, durability, and a concrete banking example. Give several paragraphs with practical details.",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path.name}")
    return value


def check_files(directory: Path, checksums: dict) -> None:
    directory = directory.resolve()
    if not isinstance(checksums, dict) or not checksums:
        raise ValueError("Expected nonempty artifact hashes")
    for name, expected in checksums.items():
        path = (directory / name).resolve()
        if not path.is_relative_to(directory) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Artifact integrity verification failed: {name}")


def verify_provenance(training_dir: Path, manifest_path: Path) -> tuple[dict, dict, dict]:
    cfg = load_json(training_dir / "resolved_config.json")
    metrics = load_json(training_dir / "metrics.json")
    manifest = load_json(manifest_path)
    if load_json(training_dir / "run_status.json").get("status") != "complete":
        raise ValueError("The source training run is not complete")
    if manifest.get("format") != "cpu-ai-lab-ollama-deployment-v1" or manifest.get("ollama_created") is not True:
        raise ValueError("A completed Ollama deployment manifest is required")
    if manifest.get("dtype") != "float16" or manifest.get("merge_dtype") != "float32":
        raise ValueError("This experiment requires the F16 export merged from FP32")
    if manifest.get("model_id") != cfg["model_id"] or manifest.get("model_revision") != cfg["revision"]:
        raise ValueError("Deployment and training source model revisions do not match")
    if manifest.get("training_metrics_sha256") != sha256(training_dir / "metrics.json"):
        raise ValueError("Deployment was not produced from these training metrics")
    adapter_dir = (training_dir / "adapter").resolve()
    declared_adapter = (manifest_path.parent / manifest["adapter_directory"]).resolve()
    if adapter_dir != declared_adapter or manifest.get("adapter_sha256") != metrics.get("artifact_sha256"):
        raise ValueError("Deployment adapter identity differs from the trained checkpoint")
    check_files(adapter_dir, metrics["artifact_sha256"])
    model_dir = (manifest_path.parent / manifest["model_directory"]).resolve()
    check_files(model_dir, manifest["model_files_sha256"])
    if manifest.get("import_format") == "gguf":
        check_files(manifest_path.parent, {manifest["gguf_file"]: manifest["gguf_sha256"]})
    test_path = training_dir / "data" / "test.jsonl"
    if sha256(test_path) != metrics["dataset"]["sha256"]["test"]:
        raise ValueError("Held-out test data changed since training")
    if not manifest.get("ollama_digest") or not manifest.get("ollama_model"):
        raise ValueError("Deployment must record an imported Ollama model and digest")
    return cfg, metrics, manifest


def make_schedule(prompts: int, repeats: int, seed: int) -> list[dict]:
    if prompts < 1 or repeats < 1:
        raise ValueError("At least one prompt and repeat is required")
    pairs = [(p, r) for r in range(repeats) for p in range(prompts)]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    offset = rng.randrange(2)
    return [{"pair_id": f"prompt-{p + 1}-repeat-{r + 1}", "prompt_index": p, "repeat": r + 1,
             "order": list(VARIANTS) if (i + offset) % 2 == 0 else list(reversed(VARIANTS))}
            for i, (p, r) in enumerate(pairs)]


def summarize_pairs(records: list[dict], requested_tokens: int) -> dict:
    grouped = {}
    for row in records:
        if row["variant"] not in VARIANTS:
            raise ValueError("Unknown variant")
        group = grouped.setdefault(row["pair_id"], {})
        if row["variant"] in group:
            raise ValueError("Duplicate backend in a timing pair")
        group[row["variant"]] = row
    included, excluded = [], []
    for pair_id, pair in grouped.items():
        if set(pair) != set(VARIANTS):
            excluded.append({"pair_id": pair_id, "reason": "incomplete pair"})
            continue
        baseline, candidate = pair[VARIANTS[0]], pair[VARIANTS[1]]
        if baseline["output_tokens"] != requested_tokens or candidate["output_tokens"] != requested_tokens:
            excluded.append({"pair_id": pair_id, "reason": "output length differs from fixed token budget",
                             "pytorch_tokens": baseline["output_tokens"], "ollama_tokens": candidate["output_tokens"]})
            continue
        hf_tokens, api_tokens = candidate.get("hf_prompt_tokens"), candidate.get("ollama_prompt_eval_count")
        if type(hf_tokens) is int and type(api_tokens) is int and abs(hf_tokens - api_tokens) > 1:
            excluded.append({"pair_id": pair_id, "reason": "prompt token counts differ by more than one; possible cache or tokenization mismatch",
                             "hf_prompt_tokens": hf_tokens, "ollama_prompt_eval_count": api_tokens})
            continue
        if any(not isinstance(r["wall_seconds"], (int, float)) or not math.isfinite(r["wall_seconds"]) or r["wall_seconds"] <= 0 for r in pair.values()):
            raise ValueError("Invalid timing")
        included.append({
            "pair_id": pair_id, "speedup_ratio": baseline["wall_seconds"] / candidate["wall_seconds"],
            "pytorch_seconds": baseline["wall_seconds"], "ollama_seconds": candidate["wall_seconds"],
            "pytorch_tokens": baseline["output_tokens"], "ollama_tokens": candidate["output_tokens"],
            "outputs_identical": baseline["output"] == candidate["output"],
            "prompt_counts_equal": candidate.get("prompt_counts_equal"),
            "hf_prompt_tokens": hf_tokens, "ollama_prompt_eval_count": api_tokens,
        })
    result = {"status": "measured" if included else "no_comparable_pairs", "included_pairs": included,
              "excluded_pairs": excluded, "complete_fixed_length_pairs": len(included),
              "requested_output_tokens": requested_tokens,
              "all_prompt_counts_equal": all(p["prompt_counts_equal"] is True for p in included) if included else None,
              "prompt_count_policy": "Pairs differing by more than one prompt token are excluded. A one-token discrepancy may be BOS/cache handling and remains explicitly flagged.",
              "comparison": "PyTorch FP32 vs Ollama/llama.cpp F16; changes runtime and weight-storage precision."}
    if included:
        ratios = [p["speedup_ratio"] for p in included]
        result.update({
            "median_paired_speedup_ratio": statistics.median(ratios),
            "min_paired_speedup_ratio": min(ratios), "max_paired_speedup_ratio": max(ratios),
            "all_included_pairs_faster": all(r > 1 for r in ratios),
            "pytorch_aggregate_tokens_per_second": len(included) * requested_tokens / sum(p["pytorch_seconds"] for p in included),
            "ollama_aggregate_tokens_per_second": len(included) * requested_tokens / sum(p["ollama_seconds"] for p in included),
        })
    return result


def quality_summary(records: list[dict]) -> dict:
    result = {}
    for variant in VARIANTS:
        rows = [r for r in records if r["variant"] == variant]
        if not rows:
            raise ValueError(f"Missing quality records for {variant}")
        result[variant] = {"examples": len(rows),
                           "json_exact_match_rate": sum(r["json_exact"] for r in rows) / len(rows),
                           "valid_json_rate": sum(r["valid_json"] for r in rows) / len(rows),
                           "text_exact_match_rate": sum(r["text_exact"] for r in rows) / len(rows)}
    result["json_exact_match_drop"] = result["pytorch_fp32"]["json_exact_match_rate"] - result["ollama_f16"]["json_exact_match_rate"]
    result["scope"] = "Descriptive results on all saved held-out extraction examples; not general quality equivalence."
    return result


def reset_ollama_runner(client: OllamaClient, model: str, digest: str, threads: int, context: int) -> dict:
    """Discard the model's KV state and reload weights without evaluating a prompt."""
    started = time.perf_counter()
    client.request("/api/generate", {"model": model, "keep_alive": 0, "stream": False,
                                     "options": {"num_gpu": 0}})
    if any(item.get("name") == model for item in client.request("/api/ps").get("models", [])):
        raise RuntimeError("Timed Ollama reset failed: the test model remained loaded after unload")
    loaded = client.request("/api/generate", {
        "model": model, "keep_alive": "10m", "stream": False,
        "options": {"num_gpu": 0, "num_thread": threads, "num_ctx": context},
    })
    status = cpu_status(client.request("/api/ps"), model, digest)
    if loaded.get("eval_count", 0) or loaded.get("prompt_eval_count", 0):
        raise RuntimeError("Model-only load unexpectedly evaluated tokens; fresh prompt state is not established")
    return {
        "mechanism": "Explicit unload, verify absence, then model-only load without a prompt",
        "wall_seconds_excluded_from_timing": time.perf_counter() - started,
        "load_response": loaded,
        "size_vram": status["size_vram"], "model_digest": status["digest"],
        "first_request_after_reload": True,
        "note": "KV state is reset; first-request kernel initialization may increase timed Ollama latency.",
    }


def run(args: argparse.Namespace) -> dict:
    if args.output.exists():
        raise FileExistsError("Choose a new result filename; existing results are not overwritten")
    for value in (args.threads, args.repeats, args.tokens, args.quality_tokens):
        if value < 1:
            raise ValueError("Thread, repeat and token counts must be positive")
    validate_config({"base_url": args.base_url, "context": args.context})
    cfg, training_metrics, manifest = verify_provenance(args.training_dir, args.deployment_manifest)
    client = OllamaClient(args.base_url, 300)
    if client.request("/api/ps").get("models"):
        raise RuntimeError("Ollama has a loaded model; run this experiment when it is idle")
    model_name, digest = manifest["ollama_model"], manifest["ollama_digest"]
    tags = [m for m in client.request("/api/tags").get("models", []) if m.get("name") == model_name]
    if len(tags) != 1 or tags[0].get("digest") != digest:
        raise ValueError("Imported Ollama model digest does not match the deployment manifest")
    if tags[0].get("details", {}).get("quantization_level") != "F16":
        raise ValueError("Imported Ollama model is not reported as F16")
    report = {
        "status": "running", "started_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "experiment": "Trained adapter CPU PyTorch FP32 vs CPU Ollama/llama.cpp F16",
        "settings": {**vars(args), "training_dir": str(args.training_dir), "deployment_manifest": str(args.deployment_manifest), "output": str(args.output)},
        "provenance": {"model_id": cfg["model_id"], "model_revision": cfg["revision"],
                       "adapter_sha256": training_metrics["artifact_sha256"], "test_data_sha256": training_metrics["dataset"]["sha256"]["test"],
                       "deployment_manifest_sha256": sha256(args.deployment_manifest), "ollama_digest": digest,
                       "gguf_sha256": manifest.get("gguf_sha256"), "import_format": manifest.get("import_format", "safetensors")},
        "ollama_model_metadata": tags[0], "ollama_version": client.request("/api/version"),
        "timing_definition": "Wall-clock generation including prefill. PyTorch inputs are pretokenized; Ollama wall time also includes HTTP and tokenization. Loading and warmup are excluded.",
        "limitations": [
            "Runtime and storage precision both change: merged FP32 PyTorch versus F16 imported by Ollama/llama.cpp.",
            "Before every timed Ollama request, the test model is unloaded, absence is verified, and a model-only request reloads it without evaluating a prompt. Reload time is excluded; first-request kernel initialization may increase timed Ollama latency. Prompt counts remain recorded because BOS/tokenization can differ.",
            "PyTorch is forced to generate the requested token count; Ollama can stop early. Any pair not reaching the exact token budget on both sides is excluded from speedup statistics.",
            "Quality evaluation is a narrow synthetic held-out task; the two arbitrary timing prompts are not a general capability evaluation.",
            "The PyTorch copy stays resident; Ollama is reloaded before each timed call. Calls run sequentially with balanced backend order. Other system activity is uncontrolled.",
            "CPU proof is a CPU-only torch build plus CPU tensors, and num_gpu=0 plus zero model VRAM in Ollama; no hardware-level GPU trace.",
        ],
        "warmups": [], "timing": [], "quality": [],
    }
    atomic_write(args.output, report)
    started_ollama = False
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        import torch
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from .training import assert_cpu_model, render_prompt
        if torch.version.cuda is not None or torch.cuda.is_available():
            raise RuntimeError("This experiment requires a CPU-only PyTorch build")
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(1)
        torch.manual_seed(args.seed)
        options = {"revision": cfg["revision"], "local_files_only": True, "trust_remote_code": False}
        tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"], **options)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(cfg["model_id"], torch_dtype=torch.float32, attn_implementation="sdpa", **options).cpu()
        model = PeftModel.from_pretrained(base, str(args.training_dir / "adapter"), is_trainable=False).merge_and_unload(safe_merge=True).cpu().float().eval()
        model.config.use_cache = True
        model.generation_config.do_sample = False
        for key in ("temperature", "top_k", "top_p"):
            setattr(model.generation_config, key, None)
        assert_cpu_model(model)
        report["versions"] = {"python": platform.python_version(), **{p: importlib.metadata.version(p) for p in ("torch", "transformers", "peft")}}
        report["cpu_evidence"] = {"torch_cuda_build": torch.version.cuda, "torch_cuda_available": torch.cuda.is_available(),
                                  "parameter_devices": sorted({str(p.device) for p in model.parameters()}), "threads": torch.get_num_threads()}

        def one(variant: str, user_prompt: str, token_limit: int, force_length: bool) -> dict:
            nonlocal started_ollama
            prompt = render_prompt(tokenizer, {"prompt": user_prompt})
            inputs = tokenizer(prompt, add_special_tokens=False, return_tensors="pt")
            inputs = {k: v.cpu() for k, v in inputs.items()}
            hf_prompt_tokens = int(inputs["input_ids"].numel())
            if variant == "pytorch_fp32":
                with torch.inference_mode():
                    start = time.perf_counter()
                    generated = model.generate(**inputs, max_new_tokens=token_limit,
                                               min_new_tokens=token_limit if force_length else 0,
                                               do_sample=False, repetition_penalty=1.0, use_cache=True,
                                               pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
                    elapsed = time.perf_counter() - start
                new_ids = generated[0, inputs["input_ids"].shape[1]:]
                return {"variant": variant, "wall_seconds": elapsed, "output_tokens": int(new_ids.numel()),
                        "output": tokenizer.decode(new_ids, skip_special_tokens=True), "hf_prompt_tokens": hf_prompt_tokens,
                        "rendered_prompt": prompt, "forced_min_new_tokens": token_limit if force_length else 0}
            started_ollama = True
            reset_record = None
            if force_length:
                reset_record = reset_ollama_runner(client, model_name, digest, args.threads, args.context)
            payload = {"model": model_name, "prompt": prompt, "raw": True, "keep_alive": "10m",
                       "options": {"num_gpu": 0, "num_thread": args.threads, "num_ctx": args.context,
                                   "num_predict": token_limit, "temperature": 0, "seed": args.seed, "repeat_penalty": 1.0}}
            response, elapsed, ttft, output, thinking = client.generate(payload)
            status = cpu_status(client.request("/api/ps"), model_name, digest)
            metrics = response_metrics(response, elapsed, ttft)
            return {"variant": variant, "wall_seconds": elapsed, "output_tokens": metrics["output_tokens"],
                    "output": output, "thinking": thinking, "hf_prompt_tokens": hf_prompt_tokens,
                    "ollama_prompt_eval_count": response.get("prompt_eval_count"),
                    "ollama_prompt_eval_cached_count": response.get("prompt_eval_cached_count"),
                    "prompt_counts_equal": response.get("prompt_eval_count") == hf_prompt_tokens,
                    "rendered_prompt": prompt, "options": payload["options"], "raw": True,
                    "cache_reset": reset_record,
                    "timing": metrics, "raw_timing_record": response,
                    "cpu_evidence": {"size_vram": status["size_vram"], "digest": status["digest"]}}

        for variant in VARIANTS:
            report["warmups"].append(one(variant, "Explain how a computer stores a number in binary.", 8, False))
            atomic_write(args.output, report)
        schedule = make_schedule(len(TIMING_PROMPTS), args.repeats, args.seed)
        report["timing_schedule"] = schedule
        for pair in schedule:
            # Change the prefix before the substantive user text for every pair.
            tagged_prompt = f"Run {pair['pair_id']}.\n" + TIMING_PROMPTS[pair["prompt_index"]]
            for variant in pair["order"]:
                row = {**one(variant, tagged_prompt, args.tokens, True), **{k: pair[k] for k in ("pair_id", "prompt_index", "repeat")}}
                report["timing"].append(row)
                atomic_write(args.output, report)
                print(json.dumps({"stage": "timing", "variant": variant, "pair": pair["pair_id"], "tokens": row["output_tokens"], "wall_tps": row["output_tokens"] / row["wall_seconds"]}), flush=True)
        report["timing_summary"] = summarize_pairs(report["timing"], args.tokens)
        test_path = args.training_dir / "data" / "test.jsonl"
        examples = [json.loads(line) for line in test_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        if len(examples) != training_metrics["dataset"]["counts"]["test"]:
            raise ValueError("Held-out example count differs from training record")
        for index, example in enumerate(examples):
            order = VARIANTS if index % 2 == 0 else tuple(reversed(VARIANTS))
            for variant in order:
                row = one(variant, example["prompt"], args.quality_tokens, False)
                row.update({"id": example["id"], "prompt": example["prompt"], "expected": example["response"],
                            **score_response(row["output"], example["response"])})
                report["quality"].append(row)
                atomic_write(args.output, report)
            print(json.dumps({"stage": "quality", "completed_examples": index + 1, "total": len(examples)}), flush=True)
        report["quality_summary"] = quality_summary(report["quality"])
        report["status"] = "complete"
    except BaseException as exc:
        report["status"] = "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        report["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if started_ollama:
            try:
                current = client.request("/api/ps").get("models", [])
                if any(m.get("name") == model_name and m.get("digest") == digest for m in current):
                    client.request("/api/generate", {"model": model_name, "keep_alive": 0, "stream": False, "options": {"num_gpu": 0}})
                report["cleanup"] = "Test model unloaded or already absent"
            except Exception as exc:
                report["cleanup"] = f"Unload failed: {exc}"
        report["finished_at_utc"] = dt.datetime.now(dt.timezone.utc).isoformat()
        atomic_write(args.output, report)
    print(json.dumps({"status": report["status"], "timing": report["timing_summary"], "quality": report["quality_summary"]}, indent=2), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=Path("artifacts/train-run"))
    parser.add_argument("--deployment-manifest", type=Path, default=Path("artifacts/deployment-f16/deployment-manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("results/backend-f16.json"))
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--quality-tokens", type=int, default=40)
    parser.add_argument("--context", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=20261025)
    parser.add_argument("--base-url", default="http://127.0.0.1:11434")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
