"""Export a hash-verified trained adapter as local FP16 safetensors for Ollama.

Export alone does not modify Ollama. Pass --create explicitly to import into the
local Ollama service; an existing model name is never intentionally overwritten.
No remote model download or model upload is performed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def local_models() -> list[dict]:
    with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=15) as response:
        return json.load(response)["models"]


def canonical_name(name: str) -> str:
    if not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*(?::[a-z0-9][a-z0-9_.-]*)?", name):
        raise ValueError("Use a local model name with an optional tag, for example cpu-ai-lab-smollm2:trained-f16.")
    return name if ":" in name else name + ":latest"


def require_unused_name(name: str) -> None:
    if any(canonical_name(entry["name"]) == name for entry in local_models() if "/" not in entry["name"]):
        raise FileExistsError(f"Ollama model {name} already exists. Choose a new name; overwrite is refused.")


def make_modelfile(tokenizer) -> tuple[str, str]:
    """Validate this model's single-turn ChatML rendering before translating it.

The benchmark sends raw prompts rendered by the original tokenizer. The explicit
Ollama template also provides matching single-turn CLI/API behavior.
"""
    marker = "CPU_AI_LAB_UNIQUE_PROMPT_MARKER_61987"
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": marker}], tokenize=False, add_generation_prompt=True,
    )
    match = re.fullmatch(
        r"<\|im_start\|>system\n(.*?)<\|im_end\|>\n<\|im_start\|>user\n"
        + marker + r"<\|im_end\|>\n<\|im_start\|>assistant\n",
        rendered,
        flags=re.DOTALL,
    )
    if not match or tokenizer.eos_token != "<|im_end|>":
        raise ValueError("This exporter requires the verified SmolLM ChatML template and EOS token.")
    system_message = match.group(1)
    if '"""' in system_message:
        raise ValueError("Unsupported triple quote in the system message.")
    go_template = (
        "{{ if .System }}<|im_start|>system\n{{ .System }}<|im_end|>\n{{ end }}"
        "{{ if .Prompt }}<|im_start|>user\n{{ .Prompt }}<|im_end|>\n{{ end }}"
        "<|im_start|>assistant\n{{ .Response }}"
    )
    modelfile = (
        "FROM ./model\n"
        + 'TEMPLATE """' + go_template + '"""\n'
        + 'SYSTEM """' + system_message + '"""\n'
        + "PARAMETER num_gpu 0\n"
        + "PARAMETER num_ctx 2048\n"
        + "PARAMETER temperature 0\n"
        + "PARAMETER repeat_penalty 1\n"
        + 'PARAMETER stop "<|im_end|>"\n'
        + 'PARAMETER stop "<|im_start|>"\n'
    )
    return modelfile, rendered


def export_safetensors(training_dir: Path, output: Path, model_name: str) -> dict:
    model_name = canonical_name(model_name)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output directory is nonempty: {output}. Use a new export directory.")
    training_dir = training_dir.resolve()
    output = output.resolve()
    metrics_path = training_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    status = json.loads((training_dir / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "complete":
        raise ValueError("Only a completed training run can be exported.")
    model_id = metrics["model"]["id"]
    revision = metrics["model"]["revision"]
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("The training model revision must be a pinned commit SHA.")
    adapter_dir = training_dir / "adapter"
    adapter_hashes = metrics["artifact_sha256"]
    if not {"adapter_model.safetensors", "adapter_config.json"} <= set(adapter_hashes):
        raise ValueError("Training metrics must record the adapter weights and configuration checksums.")
    for filename, expected in adapter_hashes.items():
        path = (adapter_dir / filename).resolve()
        if not path.is_relative_to(adapter_dir) or not path.is_file() or sha256(path) != expected:
            raise ValueError(f"Adapter checksum verification failed: {filename}")
    adapter_config = json.loads((adapter_dir / "adapter_config.json").read_text(encoding="utf-8"))
    if adapter_config.get("base_model_name_or_path") != model_id:
        raise ValueError("Adapter and training metrics reference different base models.")
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    import transformers
    import peft
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("Install the CPU-only PyTorch wheel before export.")
    torch.set_num_threads(4)
    load_options = {"revision": revision, "local_files_only": True, "trust_remote_code": False}
    tokenizer = AutoTokenizer.from_pretrained(model_id, **load_options)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    modelfile, template_probe = make_modelfile(tokenizer)
    print(json.dumps({"stage": "loading_cached_base", "model": model_id, "revision": revision}), flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_id, torch_dtype=torch.float32, attn_implementation="sdpa", use_safetensors=True, **load_options,
    ).to("cpu")
    if "LlamaForCausalLM" not in (base.config.architectures or []):
        raise ValueError("This exporter has been prepared for the LlamaForCausalLM architecture.")
    trained = PeftModel.from_pretrained(base, adapter_dir, is_trainable=False)
    merged = trained.merge_and_unload(safe_merge=True).eval()
    if {str(value.device) for value in list(merged.parameters()) + list(merged.buffers())} - {"cpu"}:
        raise RuntimeError("Export encountered non-CPU tensors.")
    merged = merged.to(dtype=torch.float16, device="cpu")
    merged.config.use_cache = True
    output.mkdir(parents=True, exist_ok=True)
    model_directory = output / "model"
    print(json.dumps({"stage": "saving_fp16_safetensors"}), flush=True)
    merged.save_pretrained(model_directory, safe_serialization=True)
    tokenizer.save_pretrained(model_directory)
    modelfile_path = output / "Modelfile"
    modelfile_path.write_text(modelfile, encoding="utf-8", newline="\n")
    manifest_path = output / "deployment-manifest.json"
    manifest = {
        "format": "cpu-ai-lab-ollama-deployment-v1",
        "exported_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "model_revision": revision,
        "adapter_directory": os.path.relpath(adapter_dir, output).replace("\\", "/"),
        "adapter_sha256": adapter_hashes,
        "training_metrics_sha256": sha256(metrics_path),
        "model_directory": "model",
        "model_files_sha256": {path.relative_to(model_directory).as_posix(): sha256(path) for path in sorted(model_directory.rglob("*")) if path.is_file()},
        "dtype": "float16",
        "merge_dtype": "float32",
        "modelfile": "Modelfile",
        "modelfile_sha256": sha256(modelfile_path),
        "template_scope": "Tokenizer-verified single-turn ChatML; benchmarks must send tokenizer-rendered prompts with raw=true.",
        "template_probe_sha256": hashlib.sha256(template_probe.encode("utf-8")).hexdigest(),
        "ollama_model": model_name,
        "ollama_created": False,
        "ollama_digest": None,
        "create_requested": False,
        "import_format": "safetensors",
        "cpu_evidence": {"torch_cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "parameter_devices": sorted({str(value.device) for value in merged.parameters()}), "parameter_dtypes": sorted({str(value.dtype) for value in merged.parameters()})},
        "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
        "notes": ["FP16 rounding is a numerical change; validate task quality before selecting this deployment.", "Export and import do not benchmark inference. Benchmark requests must force num_gpu=0 and verify zero VRAM.", "The trained adapter is merged into the original pinned base on CPU in FP32 before the FP16 cast.", "No remote download, model upload, or replacement of an existing Ollama name is requested."],
    }
    write_json(manifest_path, manifest)
    return manifest


def resume_export(training_dir: Path, output: Path, model_name: str) -> dict:
    manifest = json.loads((output / "deployment-manifest.json").read_text(encoding="utf-8"))
    if manifest.get("format") != "cpu-ai-lab-ollama-deployment-v1" or manifest.get("ollama_created"):
        raise ValueError("Resume requires a verified export that has not been imported successfully.")
    if manifest.get("ollama_model") != model_name:
        raise ValueError("The requested Ollama name differs from the saved export.")
    metrics_path = training_dir / "metrics.json"
    if sha256(metrics_path) != manifest["training_metrics_sha256"]:
        raise ValueError("Training metrics checksum changed since export.")
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    if metrics["model"]["id"] != manifest["model_id"] or metrics["model"]["revision"] != manifest["model_revision"]:
        raise ValueError("Resume model identity differs from training metrics.")
    adapter_dir = (output / manifest["adapter_directory"]).resolve()
    if adapter_dir != (training_dir / "adapter").resolve():
        raise ValueError("Resume adapter source differs from the requested training directory.")
    model_dir = (output / manifest["model_directory"]).resolve()
    if not model_dir.is_relative_to(output):
        raise ValueError("Exported model path escapes the deployment directory.")
    for directory, hashes in ((adapter_dir, manifest["adapter_sha256"]), (model_dir, manifest["model_files_sha256"])):
        for filename, expected in hashes.items():
            path = (directory / filename).resolve()
            if not path.is_relative_to(directory) or not path.is_file() or sha256(path) != expected:
                raise ValueError(f"Resume checksum verification failed: {filename}")
    if sha256(output / "Modelfile") != manifest["modelfile_sha256"]:
        raise ValueError("Modelfile checksum changed since export.")
    if manifest.get("gguf_file"):
        gguf = (output / manifest["gguf_file"]).resolve()
        if not gguf.is_relative_to(output) or sha256(gguf) != manifest["gguf_sha256"]:
            raise ValueError("Existing GGUF checksum verification failed.")
    return manifest


def convert_to_gguf(output: Path, manifest: dict, converter: Path, revision: str) -> None:
    converter = converter.resolve()
    if converter.name != "convert_hf_to_gguf.py" or not converter.is_file():
        raise ValueError("Supply the official local convert_hf_to_gguf.py script.")
    if not revision or not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("--converter-revision must specify the pinned 40-character source commit.")
    converter_hash = sha256(converter)
    source_verified = False
    if (converter.parent / ".git").exists():
        actual = subprocess.check_output(["git", "-C", str(converter.parent), "rev-parse", "HEAD"], text=True).strip()
        if actual != revision:
            raise ValueError("Converter checkout commit does not match --converter-revision.")
        source_verified = True
    gguf = output / "model-f16.gguf"
    if gguf.exists():
        if manifest.get("gguf_file") != gguf.name or sha256(gguf) != manifest.get("gguf_sha256"):
            raise FileExistsError("An unverified GGUF file already exists; refusing to overwrite it.")
        if manifest.get("converter", {}).get("sha256") != converter_hash:
            raise ValueError("The existing GGUF was made by a different converter script.")
        return
    manifest["converter"] = {
        "file": converter.name,
        "sha256": converter_hash,
        "source_commit": revision,
        "source_commit_verified": source_verified,
        "source_url": f"https://github.com/ggml-org/llama.cpp/blob/{revision}/convert_hf_to_gguf.py",
        "provenance_note": "Commit verified from local Git checkout." if source_verified else "Commit supplied for an extracted source archive; script SHA-256 records exact executed bytes.",
    }
    manifest_path = output / "deployment-manifest.json"
    write_json(manifest_path, manifest)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = ""
    started = time.perf_counter()
    try:
        subprocess.run([
            sys.executable, str(converter), str(output / manifest["model_directory"]),
            "--outfile", str(gguf), "--outtype", "f16",
        ], env=environment, check=True)
        if not gguf.is_file() or gguf.stat().st_size == 0:
            raise RuntimeError("Converter completed without producing the GGUF file.")
        manifest["gguf_file"] = gguf.name
        manifest["gguf_sha256"] = sha256(gguf)
        manifest["import_format"] = "gguf"
        modelfile_path = output / "Modelfile"
        modelfile = modelfile_path.read_text(encoding="utf-8")
        if not modelfile.startswith("FROM ./model\n"):
            raise ValueError("Unexpected source in the verified Modelfile.")
        modelfile_path.write_text(modelfile.replace("FROM ./model\n", "FROM ./model-f16.gguf\n", 1), encoding="utf-8", newline="\n")
        manifest["safetensors_modelfile_sha256"] = manifest["modelfile_sha256"]
        manifest["modelfile_sha256"] = sha256(modelfile_path)
        manifest["conversion_status"] = "complete"
    except BaseException as error:
        manifest["conversion_status"] = "failed"
        manifest["conversion_error"] = {"type": type(error).__name__, "returncode": getattr(error, "returncode", None), "message": "Conversion failed; inspect local command output. Existing files were retained."}
        raise
    finally:
        manifest["conversion_seconds"] = time.perf_counter() - started
        write_json(manifest_path, manifest)


def run(training_dir: Path, output: Path, model_name: str, create: bool = False, *, resume: bool = False, gguf_converter: Path | None = None, converter_revision: str | None = None) -> dict:
    model_name = canonical_name(model_name)
    training_dir = training_dir.resolve()
    output = output.resolve()
    if create:
        require_unused_name(model_name)
    manifest = resume_export(training_dir, output, model_name) if resume else export_safetensors(training_dir, output, model_name)
    manifest_path = output / "deployment-manifest.json"
    # Preserve the first import error from older manifests before making another attempt.
    if "import_attempts" not in manifest:
        manifest["import_attempts"] = []
        if manifest.get("create_error"):
            manifest["import_attempts"].append({"format": manifest.get("import_format", "safetensors"), "status": "failed", "error": manifest["create_error"], "seconds": manifest.get("create_seconds")})
    manifest["create_requested"] = create
    if gguf_converter:
        convert_to_gguf(output, manifest, gguf_converter, converter_revision)
    write_json(manifest_path, manifest)
    if create:
        # Recheck immediately before the mutation, after the potentially slow export.
        require_unused_name(model_name)
        environment = os.environ.copy()
        environment["OLLAMA_HOST"] = "127.0.0.1:11434"
        started = time.perf_counter()
        attempt = {"format": manifest.get("import_format", "safetensors"), "started_utc": datetime.now(timezone.utc).isoformat(), "status": "running"}
        manifest["import_attempts"].append(attempt)
        write_json(manifest_path, manifest)
        try:
            subprocess.run(["ollama", "create", model_name, "-f", "Modelfile"], cwd=output, env=environment, check=True)
            matches = [entry for entry in local_models() if entry.get("name") == model_name]
            if len(matches) != 1 or not matches[0].get("digest"):
                raise RuntimeError("Ollama create finished without a uniquely verifiable local model digest.")
            manifest["ollama_created"] = True
            manifest["ollama_digest"] = matches[0]["digest"]
            manifest["ollama_details"] = matches[0].get("details")
            manifest["create_error"] = None
            attempt["status"] = "complete"
            attempt["ollama_digest"] = matches[0]["digest"]
        except BaseException as error:
            manifest["create_error"] = {"type": type(error).__name__, "message": str(error)[:1000]}
            attempt["status"] = "failed"
            attempt["error"] = manifest["create_error"]
            raise
        finally:
            manifest["create_seconds"] = time.perf_counter() - started
            attempt["seconds"] = manifest["create_seconds"]
            write_json(manifest_path, manifest)
    print(json.dumps({"stage": "complete", "manifest": str(manifest_path), "ollama_created": manifest["ollama_created"]}), flush=True)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-dir", type=Path, default=Path("artifacts/train-run"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/deployment-f16"))
    parser.add_argument("--model", default="cpu-ai-lab-smollm2:trained-f16")
    parser.add_argument("--create", action="store_true", help="Explicitly import into local Ollama after export; refuse an existing name")
    parser.add_argument("--resume", action="store_true", help="Verify and reuse an existing export whose import did not succeed")
    parser.add_argument("--gguf-converter", type=Path, help="Path to the pinned official llama.cpp convert_hf_to_gguf.py")
    parser.add_argument("--converter-revision", "--converter-source-commit", dest="converter_revision", help="Pinned 40-character llama.cpp source commit")
    args = parser.parse_args()
    if args.gguf_converter and not args.converter_revision:
        parser.error("--gguf-converter requires --converter-revision")
    if args.converter_revision and not args.gguf_converter:
        parser.error("--converter-revision requires --gguf-converter")
    run(args.training_dir, args.output, args.model, args.create, resume=args.resume, gguf_converter=args.gguf_converter, converter_revision=args.converter_revision)


if __name__ == "__main__":
    main()
