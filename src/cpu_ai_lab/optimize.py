"""Measure CPU generation after merging LoRA, with an optional INT8 recipe.

The portable artifact is the saved adapter plus a pinned reconstruction recipe,
not a pickled model. INT8 is selected on validation quality and paired timing;
test examples are opened only after that decision has been fixed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import statistics
import time
from typing import Any


MODEL_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"
MODEL_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise ValueError(f"Empty dataset: {path}")
    return rows


def choose_subset(rows: list[Any], limit: int, seed: int) -> list[Any]:
    if limit < 1:
        raise ValueError("Sample limit must be positive")
    indices = sorted(random.Random(seed).sample(range(len(rows)), min(limit, len(rows))))
    return [rows[index] for index in indices]


def make_schedule(prompt_count: int, repeats: int, seed: int) -> list[dict[str, Any]]:
    """Randomized paired order, with AB and BA alternated across pairs."""
    if prompt_count < 1 or repeats < 1:
        raise ValueError("At least one prompt and repeat are required")
    pairs = [(prompt, repeat) for repeat in range(repeats) for prompt in range(prompt_count)]
    rng = random.Random(seed)
    rng.shuffle(pairs)
    initial = rng.randrange(2)
    return [
        {
            "pair": pair,
            "prompt_index": prompt,
            "repeat": repeat,
            "order": ["fp32", "int8"] if (pair + initial) % 2 == 0 else ["int8", "fp32"],
        }
        for pair, (prompt, repeat) in enumerate(pairs)
    ]


def summarize_timings(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        raise ValueError("Timing records are empty")
    grouped = {name: [row for row in records if row["variant"] == name] for name in ("fp32", "int8")}
    if any(not rows for rows in grouped.values()):
        raise ValueError("Both FP32 and INT8 timings are required")
    summary: dict[str, Any] = {}
    for name, rows in grouped.items():
        if any(row["seconds"] <= 0 or row["output_tokens"] < 1 for row in rows):
            raise ValueError("Timings and token counts must be positive")
        elapsed = sum(row["seconds"] for row in rows)
        tokens = sum(row["output_tokens"] for row in rows)
        summary[name] = {
            "calls": len(rows),
            "total_output_tokens": tokens,
            "total_generation_seconds": elapsed,
            "aggregate_tokens_per_second": tokens / elapsed,
            "median_call_seconds": statistics.median(row["seconds"] for row in rows),
        }
    pairs: dict[int, dict[str, dict[str, Any]]] = {}
    for row in records:
        pair = pairs.setdefault(row["pair"], {})
        if row["variant"] in pair:
            raise ValueError("Duplicate variant in a benchmark pair")
        pair[row["variant"]] = row
    ratios = []
    for pair in pairs.values():
        if set(pair) != {"fp32", "int8"}:
            raise ValueError("Incomplete benchmark pair")
        if pair["fp32"]["output_tokens"] != pair["int8"]["output_tokens"]:
            raise ValueError("Paired generation must emit the same token count")
        ratios.append(pair["fp32"]["seconds"] / pair["int8"]["seconds"])
    summary["int8_speedup_ratio"] = summary["int8"]["aggregate_tokens_per_second"] / summary["fp32"]["aggregate_tokens_per_second"]
    summary["paired_speedup_ratios"] = ratios
    summary["paired_median_speedup_ratio"] = statistics.median(ratios)
    summary["paired_min_speedup_ratio"] = min(ratios)
    summary["paired_max_speedup_ratio"] = max(ratios)
    return summary


def quality_gate(baseline: dict[str, Any], candidate: dict[str, Any], max_loss_ratio: float = 1.10, max_loss_increase: float = 0.0) -> dict[str, Any]:
    if not math.isfinite(max_loss_ratio) or max_loss_ratio < 1 or not math.isfinite(max_loss_increase) or max_loss_increase < 0:
        raise ValueError("Loss ratio must be finite and at least one; absolute increase must be finite and nonnegative")
    base_loss = float(baseline["loss"]["loss"])
    candidate_loss = float(candidate["loss"]["loss"])
    losses_valid = math.isfinite(base_loss) and math.isfinite(candidate_loss) and base_loss > 0 and candidate_loss >= 0
    allowed_loss = max(base_loss * max_loss_ratio, base_loss + max_loss_increase) if losses_valid else None
    checks = {
        "finite_losses": losses_valid,
        "loss_within_tolerance": losses_valid and candidate_loss <= allowed_loss,
        "no_json_exact_match_drop": candidate["generation"]["json_exact_match_rate"] >= baseline["generation"]["json_exact_match_rate"],
        "no_valid_json_drop": candidate["generation"]["valid_json_rate"] >= baseline["generation"]["valid_json_rate"],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "max_loss_ratio": max_loss_ratio,
        "max_loss_increase": max_loss_increase,
        "allowed_candidate_loss": allowed_loss,
        "loss_policy": "candidate response-token NLL <= max(baseline NLL * max_loss_ratio, baseline NLL + max_loss_increase); exact-match and valid-JSON rates must not decrease",
        "observed_loss_ratio": candidate_loss / base_loss if losses_valid else None,
        "scope": "Predeclared guard on this validation subset; not statistical equivalence or general quality preservation.",
    }


def select_linear_names(names: list[str], scope: str) -> list[str]:
    """Choose named float Linear modules; MLP-only preserves attention/head."""
    if scope == "all-linear":
        return list(names)
    if scope == "mlp-only":
        suffixes = (".gate_proj", ".up_proj", ".down_proj")
        return [name for name in names if ".mlp." in name and name.endswith(suffixes)]
    raise ValueError(f"Unknown quantization scope: {scope}")


def quantize_cpu_model(model: Any, scope: str, inplace: bool = False):
    import torch
    names = select_linear_names([name for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)], scope)
    if not names:
        raise ValueError(f"No supported linear modules found for scope {scope}")
    quantized = torch.ao.quantization.quantize_dynamic(model, set(names), dtype=torch.qint8, inplace=inplace)
    return quantized.eval(), names


def select_variant(gate: dict[str, Any], timing: dict[str, Any], minimum_speedup: float = 1.03) -> dict[str, Any]:
    speed_pass = timing["int8_speedup_ratio"] >= minimum_speedup and timing["paired_min_speedup_ratio"] > 1.0
    return {
        "variant": "int8" if gate["passed"] and speed_pass else "fp32",
        "quality_gate_passed": bool(gate["passed"]),
        "speed_gate_passed": speed_pass,
        "minimum_aggregate_speedup": minimum_speedup,
        "requires_all_measured_pairs_faster": True,
        "selection_data": "validation subset and validation timing prompts only",
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def load_recipe(recipe_path: str | Path):
    """Reconstruct the selected local model from a hash-checked adapter.

    Returns ``(model, tokenizer, recipe)``. The recipe is JSON and the model
    weights are safetensors; no arbitrary Python object is deserialized.
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import transformers
    import peft
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    recipe_path = Path(recipe_path)
    recipe = json.loads(recipe_path.read_text(encoding="utf-8"))
    if recipe.get("format") != "cpu-ai-lab-reconstruction-v1" or recipe.get("selected_variant") not in ("fp32", "int8"):
        raise ValueError("Unsupported reconstruction recipe")
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("Install the CPU-only PyTorch wheel before loading this recipe")
    installed = {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}
    for package, version in installed.items():
        if version != recipe["versions"][package]:
            raise RuntimeError(f"Reproduction requires {package}=={recipe['versions'][package]}, found {version}")
    torch.set_num_threads(int(recipe["threads"]))
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    adapter_dir = (recipe_path.parent / recipe["adapter_directory"]).resolve()
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError("The recipe requires an adapter_model.safetensors checkpoint")
    for filename, expected_hash in recipe["adapter_sha256"].items():
        path = (adapter_dir / filename).resolve()
        if not path.is_relative_to(adapter_dir) or sha256(path) != expected_hash:
            raise ValueError(f"Adapter integrity verification failed: {filename}")
    options = {"revision": recipe["model_revision"], "local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(recipe["model_id"], **options)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        recipe["model_id"], torch_dtype=torch.float32, attn_implementation="sdpa", **options,
    ).cpu()
    model = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False).merge_and_unload(safe_merge=True).cpu().float().eval()
    model.config.use_cache = True
    model.generation_config.do_sample = False
    for name in ("temperature", "top_p", "top_k"):
        setattr(model.generation_config, name, None)
    if recipe["selected_variant"] == "int8":
        engine = recipe["quantization_engine"]
        if engine not in torch.backends.quantized.supported_engines:
            raise RuntimeError(f"Quantization engine is unavailable: {engine}")
        torch.backends.quantized.engine = engine
        model, _ = quantize_cpu_model(model, recipe.get("quantization_scope", "all-linear"), inplace=True)
    return model, tokenizer, recipe


def run(args: argparse.Namespace) -> dict[str, Any]:
    # Explicitly suppress GPU visibility before importing the learning framework.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import transformers
    import peft
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from cpu_ai_lab.training import encode_examples, evaluate_generation, evaluate_loss, render_prompt

    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("This experiment requires the CPU-only PyTorch wheel, not a CUDA build")
    torch.set_num_threads(args.threads)
    torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed)
    engines = list(torch.backends.quantized.supported_engines)
    engine = next((value for value in ("x86", "fbgemm") if value in engines), None)
    if engine is None:
        raise RuntimeError(f"No tested x86 quantization engine is available: {engines}")
    torch.backends.quantized.engine = engine

    training_dir = Path(args.training_dir)
    output = Path(args.output)
    recipe_path = output.with_name(output.stem + "-recipe.json")
    if output.exists() or recipe_path.exists():
        raise FileExistsError("Choose a new output filename; existing experiment results are not overwritten")
    config = json.loads((training_dir / "resolved_config.json").read_text(encoding="utf-8"))
    model_id = config.get("model_id", MODEL_ID)
    revision = config.get("revision", config.get("model_revision", MODEL_REVISION))
    max_length = int(config.get("max_sequence_length", config.get("max_length", 256)))
    validation_path = training_dir / "data" / "validation.jsonl"
    validation_all = read_jsonl(validation_path)
    validation = choose_subset(validation_all, args.quality_examples, args.seed)
    adapter_dir = training_dir / "adapter"
    adapter_files = sorted(path for path in adapter_dir.rglob("*") if path.is_file())
    if not adapter_files:
        raise FileNotFoundError(f"No trained adapter in {adapter_dir}")

    report: dict[str, Any] = {
        "status": "running",
        "experiment": "CPU merged-adapter FP32 versus dynamically quantized INT8 linear layers",
        "quantization_scope": args.quantization_scope,
        "model_id": model_id,
        "model_revision": revision,
        "adapter_sha256": {path.relative_to(adapter_dir).as_posix(): sha256(path) for path in adapter_files},
        "versions": {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "cpu_only": {"torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "parameter_devices": []},
        "settings": vars(args),
        "quantization_engine": engine,
        "validation": {"available_examples": len(validation_all), "evaluated_examples": len(validation), "dataset_sha256": sha256(validation_path)},
        "timing_definition": "Wall-clock model.generate on pretokenized batch-one inputs, including prompt processing and generated tokens; excludes tokenization, loading, conversion and warmup.",
        "limitations": [
            "INT8 applies only to the explicitly recorded linear modules; all other modules remain floating point. MLP-only preserves attention projections and the language-model output head.",
            "Fixed-token timing suppresses early stopping to make work comparable; quality evaluation separately allows normal stopping.",
            "Validation/test subsets and repeated short calls are small; results do not establish broad quality equivalence.",
            "Both models are resident during timings; this experiment does not claim per-variant peak RAM savings.",
            "The original pretrained base was not trained by this project; this experiment uses the project's CPU-trained LoRA adapter.",
        ],
        "benchmark_records": [],
    }
    write_json(output, report)
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision, local_files_only=True, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.float32,
        local_files_only=True, trust_remote_code=False, attn_implementation="sdpa",
    ).cpu()
    fp32 = PeftModel.from_pretrained(base, str(adapter_dir), is_trainable=False).merge_and_unload(safe_merge=True).cpu().float().eval()
    fp32.config.use_cache = True
    # Remove sampling defaults from the checkpoint for deterministic greedy calls.
    for name in ("temperature", "top_p", "top_k"):
        setattr(fp32.generation_config, name, None)
    fp32.generation_config.do_sample = False
    started = time.perf_counter()
    int8, quantized_names = quantize_cpu_model(fp32, args.quantization_scope, inplace=False)
    report["quantization_seconds"] = time.perf_counter() - started
    report["quantized_linear_modules"] = sum(isinstance(module, torch.ao.nn.quantized.dynamic.Linear) for module in int8.modules())
    report["quantized_module_names"] = quantized_names
    if not report["quantized_linear_modules"]:
        raise RuntimeError("No linear layers were quantized")
    models = {"fp32": fp32, "int8": int8}
    devices = sorted({str(tensor.device) for model in models.values() for tensor in list(model.parameters()) + list(model.buffers())})
    if devices != ["cpu"]:
        raise RuntimeError(f"Unexpected device allocation: {devices}")
    report["cpu_only"]["parameter_devices"] = devices
    encoded = encode_examples(tokenizer, validation, max_length)
    for name, model in models.items():
        print(f"Validating {name} on {len(validation)} examples", flush=True)
        report["validation"][name] = {
            "loss": evaluate_loss(model, encoded, tokenizer.pad_token_id, batch_size=2),
            "generation": evaluate_generation(model, tokenizer, validation, max_new_tokens=args.quality_tokens),
        }
        write_json(output, report)
    gate = quality_gate(report["validation"]["fp32"], report["validation"]["int8"], args.max_loss_ratio, args.max_loss_increase)
    report["validation_quality_gate"] = gate

    timing_examples = validation[:min(args.prompts, len(validation))]
    inputs = [tokenizer(render_prompt(tokenizer, row), return_tensors="pt", add_special_tokens=False) for row in timing_examples]

    def generate_fixed(model: Any, input_dict: Any, tokens: int) -> tuple[float, int, str]:
        with torch.inference_mode():
            started = time.perf_counter()
            generated = model.generate(
                **input_dict, do_sample=False, min_new_tokens=tokens, max_new_tokens=tokens,
                use_cache=True, pad_token_id=tokenizer.pad_token_id,
            )
            elapsed = time.perf_counter() - started
        token_ids = generated[0, input_dict["input_ids"].shape[1]:]
        if len(token_ids) != tokens:
            raise RuntimeError(f"Expected {tokens} output tokens, got {len(token_ids)}")
        return elapsed, len(token_ids), tokenizer.decode(token_ids, skip_special_tokens=True)

    for model in models.values():
        for _ in range(args.warmups):
            generate_fixed(model, inputs[0], min(16, args.tokens))
    schedule = make_schedule(len(inputs), args.repeats, args.seed)
    report["benchmark_schedule"] = schedule
    for pair in schedule:
        for name in pair["order"]:
            elapsed, count, generated = generate_fixed(models[name], inputs[pair["prompt_index"]], args.tokens)
            record = {
                "pair": pair["pair"], "repeat": pair["repeat"], "prompt_index": pair["prompt_index"],
                "variant": name, "seconds": elapsed, "output_tokens": count,
                "prompt_tokens": int(inputs[pair["prompt_index"]]["input_ids"].shape[1]),
                "tokens_per_second": count / elapsed, "output": generated,
            }
            report["benchmark_records"].append(record)
            write_json(output, report)
            print(f"pair={pair['pair']} {name}: {count / elapsed:.2f} generated tokens/s (includes prompt processing)", flush=True)
    timing = summarize_timings(report["benchmark_records"])
    report["timing_summary"] = timing
    decision = select_variant(gate, timing, args.minimum_speedup)
    report["selection"] = decision
    write_json(output, report)

    # Nothing from test influences model/quantization selection or thresholds.
    test_path = training_dir / "data" / "test.jsonl"
    test_all = read_jsonl(test_path)
    test = choose_subset(test_all, args.quality_examples, args.seed + 1)
    test_encoded = encode_examples(tokenizer, test, max_length)
    report["test"] = {"available_examples": len(test_all), "evaluated_examples": len(test), "dataset_sha256": sha256(test_path)}
    for name in dict.fromkeys(["fp32", decision["variant"]]):
        model = models[name]
        print(f"Final test evaluation: {name}", flush=True)
        report["test"][name] = {
            "loss": evaluate_loss(model, test_encoded, tokenizer.pad_token_id, batch_size=2),
            "generation": evaluate_generation(model, tokenizer, test, max_new_tokens=args.quality_tokens),
        }
        write_json(output, report)
    recipe = {
        "format": "cpu-ai-lab-reconstruction-v1",
        "selected_variant": decision["variant"],
        "model_id": model_id, "model_revision": revision,
        "adapter_directory": os.path.relpath(adapter_dir, recipe_path.parent).replace("\\", "/"),
        "adapter_sha256": report["adapter_sha256"],
        "versions": report["versions"],
        "threads": args.threads, "inter_op_threads": 1,
        "quantization_engine": engine if decision["variant"] == "int8" else None,
        "quantization_scope": args.quantization_scope if decision["variant"] == "int8" else None,
        "quantized_module_names": quantized_names if decision["variant"] == "int8" else [],
        "steps": [
            "Load the pinned base checkpoint locally in FP32 on CPU.",
            "Load the saved adapter with PeftModel.from_pretrained and merge_and_unload(safe_merge=True).",
            "Set eval mode and use_cache=True.",
        ] + (["Apply dynamic INT8 quantization to the exact named linear modules selected by quantization_scope and listed in quantized_module_names."] if decision["variant"] == "int8" else []),
        "serialization": "Reconstruct from safetensors adapter and pinned base; no pickled executable model is required.",
    }
    write_json(recipe_path, recipe)
    report["recipe_file"] = recipe_path.name
    report["status"] = "complete"
    write_json(output, report)
    print(json.dumps({"selected": decision["variant"], "int8_speedup": timing["int8_speedup_ratio"], "validation_quality_pass": gate["passed"], "results": str(output)}), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--load-recipe", help="Load an existing measured recipe instead of running a benchmark")
    parser.add_argument("--prompt", help="User prompt for --load-recipe")
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--training-dir", default="artifacts/train-run")
    parser.add_argument("--output", default="results/optimization.json")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--quality-examples", type=int, default=8)
    parser.add_argument("--quality-tokens", type=int, default=40)
    parser.add_argument("--prompts", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2718)
    parser.add_argument("--max-loss-ratio", type=float, default=1.10)
    parser.add_argument("--max-loss-increase", type=float, default=0.0, help="Explicit allowed absolute response-token NLL increase; combines with the relative tolerance using max")
    parser.add_argument("--quantization-scope", choices=("all-linear", "mlp-only"), default="all-linear")
    parser.add_argument("--minimum-speedup", type=float, default=1.03)
    args = parser.parse_args()
    if args.load_recipe:
        if not args.prompt or args.max_new_tokens < 1:
            parser.error("--load-recipe requires --prompt and positive --max-new-tokens")
        model, tokenizer, recipe = load_recipe(args.load_recipe)
        import torch
        text = tokenizer.apply_chat_template([{"role": "user", "content": args.prompt}], tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt", add_special_tokens=False)
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False, use_cache=True, pad_token_id=tokenizer.pad_token_id)
        print(tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True))
        return
    if args.prompt:
        parser.error("--prompt is only valid with --load-recipe")
    for name in ("threads", "quality_examples", "quality_tokens", "prompts", "tokens", "repeats", "warmups"):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if not math.isfinite(args.max_loss_ratio) or args.max_loss_ratio < 1 or not math.isfinite(args.minimum_speedup) or args.minimum_speedup < 1:
        parser.error("Quality loss ratio and minimum speedup must be finite and at least one")
    if not math.isfinite(args.max_loss_increase) or args.max_loss_increase < 0:
        parser.error("Maximum loss increase must be finite and nonnegative")
    run(args)


if __name__ == "__main__":
    main()
