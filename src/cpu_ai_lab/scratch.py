"""Train a small byte-level causal transformer from random weights on CPU.

No pretrained model or tokenizer is loaded. The fixed 256-symbol byte vocabulary
does not inspect held-out text. Loss/perplexity are per byte, not per subword.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import threading
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import psutil
import torch
from safetensors.torch import load_file, save_file
from torch import nn
from torch.nn import functional as F


DATA_REVISION = "6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e"
DATA_URL = f"https://raw.githubusercontent.com/karpathy/char-rnn/{DATA_REVISION}/data/tinyshakespeare/input.txt"
DATA_SHA256 = "86c4e6aa9db7c042ec79f339dcb96d42b0075e16b8fc2e86bf0ca57e2dc565ed"


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def state_checksum(model: nn.Module) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def assert_cpu(model: nn.Module) -> None:
    devices = {str(t.device) for t in list(model.parameters()) + list(model.buffers())}
    if devices != {"cpu"}:
        raise RuntimeError(f"Expected CPU-only tensors, got {sorted(devices)}.")


class PeakRSS:
    def __init__(self):
        self.peak_bytes = psutil.Process().memory_info().rss
        self.event = threading.Event()
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self):
        process = psutil.Process()
        while not self.event.is_set():
            self.peak_bytes = max(self.peak_bytes, process.memory_info().rss)
            self.event.wait(0.05)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self, *_):
        self.peak_bytes = max(self.peak_bytes, psutil.Process().memory_info().rss)
        self.event.set()
        self.thread.join()


def get_corpus(path: Path, allow_download: bool = True) -> bytes:
    if not path.is_file():
        if not allow_download:
            raise FileNotFoundError(f"Corpus is not cached at {path}.")
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(DATA_URL, headers={"User-Agent": "cpu-only-ai-lab"})
        with urllib.request.urlopen(request, timeout=60) as response:
            payload = response.read(2_000_000)
        if sha256_bytes(payload) != DATA_SHA256:
            raise ValueError("Downloaded corpus checksum does not match the pinned source.")
        path.write_bytes(payload)
    payload = path.read_bytes()
    if sha256_bytes(payload) != DATA_SHA256:
        raise ValueError("Cached corpus checksum does not match the pinned source.")
    return payload


def split_corpus(payload: bytes, train_fraction: float = 0.9, validation_fraction: float = 0.05) -> tuple[dict, dict]:
    if not (0 < train_fraction < 1 and 0 < validation_fraction < 1 and train_fraction + validation_fraction < 1):
        raise ValueError("Train/validation fractions must leave three positive splits.")
    first = int(len(payload) * train_fraction)
    second = first + int(len(payload) * validation_fraction)
    bounds = {"train": (0, first), "validation": (first, second), "test": (second, len(payload))}
    splits, metadata = {}, {}
    for name, (start, stop) in bounds.items():
        segment = payload[start:stop]
        if len(segment) < 2:
            raise ValueError("Every split needs at least two bytes.")
        splits[name] = torch.tensor(list(segment), dtype=torch.long, device="cpu")
        metadata[name] = {"start_byte_inclusive": start, "stop_byte_exclusive": stop, "bytes": len(segment), "sha256": sha256_bytes(segment)}
    return splits, metadata


def random_batch(segment: torch.Tensor, batch_size: int, block_size: int, generator: torch.Generator) -> tuple[torch.Tensor, torch.Tensor]:
    if segment.device.type != "cpu" or segment.numel() <= block_size:
        raise ValueError("Batch source must be a sufficiently long CPU tensor.")
    starts = torch.randint(0, segment.numel() - block_size, (batch_size,), generator=generator, device="cpu")
    offsets = starts[:, None] + torch.arange(block_size, device="cpu")[None, :]
    return segment[offsets], segment[offsets + 1]


class CausalBlock(nn.Module):
    def __init__(self, width: int, heads: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.dropout = dropout
        self.ln_attention = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.projection = nn.Linear(width, width)
        self.ln_mlp = nn.LayerNorm(width)
        self.mlp = nn.Sequential(nn.Linear(width, width * 4), nn.GELU(), nn.Linear(width * 4, width), nn.Dropout(dropout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, width = x.shape
        q, k, v = self.qkv(self.ln_attention(x)).chunk(3, dim=-1)
        q, k, v = (t.reshape(batch, length, self.heads, width // self.heads).transpose(1, 2) for t in (q, k, v))
        attention = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        attention = attention.transpose(1, 2).contiguous().reshape(batch, length, width)
        x = x + F.dropout(self.projection(attention), p=self.dropout, training=self.training)
        return x + self.mlp(self.ln_mlp(x))


class ByteTransformer(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        width, heads = int(config["width"]), int(config["heads"])
        self.block_size = int(config["block_size"])
        if width < 1 or heads < 1 or width % heads or self.block_size < 1 or int(config["layers"]) < 1:
            raise ValueError("Invalid model dimensions.")
        self.embedding = nn.Embedding(256, width)
        self.position = nn.Embedding(self.block_size, width)
        self.blocks = nn.ModuleList([CausalBlock(width, heads, config["dropout"]) for _ in range(config["layers"])])
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 256, bias=False)
        self.apply(self._initialize)

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.block_size:
            raise ValueError("Input must be a nonempty batch no longer than block_size.")
        positions = torch.arange(ids.shape[1], device="cpu")
        x = self.embedding(ids) + self.position(positions)
        for block in self.blocks:
            x = block(x)
        return self.output(self.norm(x))


@torch.inference_mode()
def evaluate(model: ByteTransformer, segment: torch.Tensor, batch_size: int = 16) -> dict:
    """Score each next-byte target once; context resets every block_size bytes."""
    if segment.numel() < 2 or batch_size < 1:
        raise ValueError("Evaluation needs at least two bytes and positive batch size.")
    model.eval()
    total_nll, total_tokens = 0.0, 0
    block_size = model.block_size
    started = time.perf_counter()
    starts = list(range(0, segment.numel() - 1, block_size))
    for offset in range(0, len(starts), batch_size):
        batch_starts = starts[offset:offset + batch_size]
        x = torch.zeros((len(batch_starts), block_size), dtype=torch.long, device="cpu")
        y = torch.full_like(x, -100)
        for row, start in enumerate(batch_starts):
            length = min(block_size, segment.numel() - start - 1)
            x[row, :length] = segment[start:start + length]
            y[row, :length] = segment[start + 1:start + length + 1]
        logits = model(x)
        total_nll += float(F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1), reduction="sum"))
        total_tokens += int((y != -100).sum())
    loss = total_nll / total_tokens
    if not math.isfinite(loss):
        raise RuntimeError("Nonfinite evaluation loss.")
    return {"loss_nats_per_byte": loss, "bits_per_byte": loss / math.log(2), "byte_perplexity": math.exp(loss), "target_bytes": total_tokens, "seconds": time.perf_counter() - started, "context_policy": "reset every block_size; each next-byte target scored once; last block masked"}


@torch.inference_mode()
def generate(model: ByteTransformer, prompt: str, new_bytes: int, temperature: float, top_k: int, seed: int) -> dict:
    if not prompt or new_bytes < 1 or temperature <= 0 or not 1 <= top_k <= 256:
        raise ValueError("Invalid generation settings.")
    model.eval()
    generator = torch.Generator(device="cpu").manual_seed(seed)
    ids = torch.tensor([list(prompt.encode("utf-8"))], dtype=torch.long, device="cpu")
    prompt_size = ids.shape[1]
    started = time.perf_counter()
    for _ in range(new_bytes):
        logits = model(ids[:, -model.block_size:])[:, -1, :] / temperature
        threshold = torch.topk(logits, top_k).values[:, -1, None]
        logits = logits.masked_fill(logits < threshold, float("-inf"))
        next_id = torch.multinomial(torch.softmax(logits, dim=-1), num_samples=1, generator=generator)
        ids = torch.cat((ids, next_id), dim=1)
    seconds = time.perf_counter() - started
    output = bytes(ids[0, prompt_size:].tolist())
    return {"prompt": prompt, "generated": output.decode("utf-8", errors="replace"), "generated_hex": output.hex(), "new_bytes": new_bytes, "seconds": seconds, "bytes_per_second_including_prompt": new_bytes / seconds, "temperature": temperature, "top_k": top_k, "seed": seed}


def unigram_baseline(train: torch.Tensor, test: torch.Tensor) -> dict:
    counts = torch.bincount(train, minlength=256).double() + 1
    log_prob = torch.log(counts / counts.sum())
    loss = float(-log_prob[test[1:]].mean())
    return {"loss_nats_per_byte": loss, "byte_perplexity": math.exp(loss), "target_bytes": test.numel() - 1, "smoothing": "add-one; fitted on train bytes only"}


def _run(config: dict, output: Path, corpus_path: Path, allow_download: bool) -> dict:
    for key in ("steps", "threads", "batch_size", "evaluation_batch_size", "validation_every", "sample_bytes", "block_size"):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive.")
    if not 0 <= config["dropout"] < 1 or config["learning_rate"] <= 0 or config["max_train_seconds"] <= 0:
        raise ValueError("Invalid dropout, learning rate, or time budget.")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.set_num_threads(config["threads"])
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    if torch.version.cuda is not None or torch.cuda.is_available():
        raise RuntimeError("Install the CPU-only PyTorch wheel for this experiment.")
    write_json(output / "resolved_config.json", config)
    started = time.perf_counter()
    with PeakRSS() as memory:
        payload = get_corpus(corpus_path, allow_download)
        splits, boundaries = split_corpus(payload, config["train_fraction"], config["validation_fraction"])
        if any(segment.numel() <= config["block_size"] for segment in splits.values()):
            raise ValueError("Each split must be longer than block_size.")
        provenance = {"source_url": DATA_URL, "source_revision": DATA_REVISION, "sha256": DATA_SHA256, "total_bytes": len(payload), "dataset": "tiny Shakespeare: subset of Shakespeare works distributed in karpathy/char-rnn", "license_note": "Upstream repository README declares MIT; underlying Shakespeare works are public domain. Corpus kept local; no upstream code copied.", "license_source": f"https://github.com/karpathy/char-rnn/blob/{DATA_REVISION}/Readme.md#license", "splits": boundaries, "split_policy": "Contiguous source-byte intervals; train/validation/test never share source positions; no batch crosses a split. Natural repeated phrases may occur across the corpus.", "tokenization": "fixed 256-byte vocabulary; no tokenizer fitting"}
        write_json(output / "data-provenance.json", provenance)
        model = ByteTransformer(config).to(device="cpu", dtype=torch.float32)
        assert_cpu(model)
        initial_checksum = state_checksum(model)
        parameter_count = sum(p.numel() for p in model.parameters())
        print(json.dumps({"stage": "baseline", "parameters": parameter_count, "source_bytes": len(payload)}), flush=True)
        baseline = {"validation": evaluate(model, splits["validation"], config["evaluation_batch_size"]), "test": evaluate(model, splits["test"], config["evaluation_batch_size"]), "test_unigram": unigram_baseline(splits["train"], splits["test"]), "sample": generate(model, config["sample_prompt"], config["sample_bytes"], config["temperature"], config["top_k"], config["seed"] + 7)}
        write_json(output / "baseline.json", baseline)
        best_loss, best_step = baseline["validation"]["loss_nats_per_byte"], 0
        best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
        optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"], weight_decay=config["weight_decay"])
        train_generator = torch.Generator(device="cpu").manual_seed(config["seed"] + 1)
        curve, validation_history = [], [{"step": 0, **baseline["validation"]}]
        training_started, compute_seconds, total_bytes = time.perf_counter(), 0.0, 0
        completed_step = 0
        stop_reason = "step_budget"
        for step in range(1, config["steps"] + 1):
            step_started = time.perf_counter()
            model.train()
            x, y = random_batch(splits["train"], config["batch_size"], config["block_size"], train_generator)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x)
            loss = F.cross_entropy(logits.reshape(-1, 256), y.reshape(-1))
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite training loss.")
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), config["gradient_clip"]))
            optimizer.step()
            seconds = time.perf_counter() - step_started
            compute_seconds += seconds
            total_bytes += y.numel()
            completed_step = step
            curve.append({"step": step, "train_loss_nats_per_byte": float(loss.detach()), "gradient_norm_before_clip": norm, "step_compute_seconds": seconds, "target_bytes": y.numel()})
            budget_reached = time.perf_counter() - training_started >= config["max_train_seconds"]
            if step % config["validation_every"] == 0 or step == config["steps"] or budget_reached:
                validation = evaluate(model, splits["validation"], config["evaluation_batch_size"])
                validation_history.append({"step": step, **validation})
                if validation["loss_nats_per_byte"] < best_loss:
                    best_loss, best_step = validation["loss_nats_per_byte"], step
                    best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
                progress = {"stage": "training", "step": step, "train_loss": curve[-1]["train_loss_nats_per_byte"], "validation_loss": validation["loss_nats_per_byte"], "best_step": best_step, "elapsed_seconds": time.perf_counter() - training_started}
                print(json.dumps(progress), flush=True)
                write_json(output / "learning-curve.json", {"steps": curve, "validation": validation_history})
                write_json(output / "run_status.json", {"status": "running", **progress})
            if budget_reached:
                stop_reason = "wall_time_budget_checked_after_step"
                break
        train_seconds = time.perf_counter() - training_started
        model.load_state_dict(best_state)
        assert_cpu(model)
        final_checksum = state_checksum(model)
        after = {"validation": evaluate(model, splits["validation"], config["evaluation_batch_size"]), "test": evaluate(model, splits["test"], config["evaluation_batch_size"]), "sample": generate(model, config["sample_prompt"], config["sample_bytes"], config["temperature"], config["top_k"], config["seed"] + 7)}
        checkpoint = output / "model.safetensors"
        save_file({name: value.contiguous() for name, value in best_state.items()}, str(checkpoint), metadata={"format": "cpu-ai-lab-byte-transformer-v1", "initialization": "random", "seed": str(config["seed"])})
        write_json(output / "model-config.json", config)
        # Reopen the artifact without pickle and check its exact parameter bytes.
        reloaded = ByteTransformer(config).to("cpu")
        reloaded.load_state_dict(load_file(str(checkpoint), device="cpu"))
        assert_cpu(reloaded)
        reload_checksum = state_checksum(reloaded)
        if reload_checksum != final_checksum:
            raise RuntimeError("Saved model failed exact-state reload verification.")
        metrics = {"schema_version": 1, "experiment": "CPU-only byte transformer trained from random weights", "timestamp_utc": datetime.now(timezone.utc).isoformat(), "config": config, "data": provenance, "model": {"parameters": parameter_count, "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad), "pretrained_weights_loaded": False, "initialization_seed": config["seed"], "initial_state_sha256": initial_checksum, "selected_state_sha256": final_checksum, "parameters_changed": initial_checksum != final_checksum, "checkpoint_sha256": sha256_bytes(checkpoint.read_bytes()), "checkpoint_bytes": checkpoint.stat().st_size, "reload_state_sha256": reload_checksum, "reload_verified": True, "architecture": "pre-norm causal transformer, untied input/output embeddings, learned positions, GELU, SDPA; fixed byte vocabulary256"}, "before": baseline, "after": after, "selection": {"criterion": "lowest full validation loss, including random initial model", "best_step": best_step, "best_validation_loss": best_loss, "test_used_for_selection": False}, "training": {"completed_steps": completed_step, "requested_steps": config["steps"], "stop_reason": stop_reason, "target_bytes_processed": total_bytes, "compute_seconds": compute_seconds, "loop_seconds_including_validation": train_seconds, "compute_target_bytes_per_second": total_bytes / compute_seconds, "wall_target_bytes_per_second": total_bytes / train_seconds}, "cpu_evidence": {"torch_version": torch.__version__, "cuda_build": torch.version.cuda, "cuda_available": torch.cuda.is_available(), "model_devices": sorted({str(p.device) for p in model.parameters()}), "optimizer_tensor_devices": sorted({str(value.device) for state in optimizer.state.values() for value in state.values() if isinstance(value, torch.Tensor)}), "dtype": "float32", "threads": torch.get_num_threads(), "interop_threads": torch.get_num_interop_threads(), "logical_cpu_count": psutil.cpu_count(), "physical_cpu_count": psutil.cpu_count(logical=False), "platform": platform.system(), "python_version": platform.python_version(), "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}, "limits": ["One short run on a tiny Shakespeare corpus, not a general assistant.", "Byte perplexity cannot be compared directly to subword-token perplexity.", "Contiguous source splits avoid overlapping positions, but natural repeated phrases remain.", "Memory is sampled process RSS, not an exact peak or whole-machine energy measurement.", "Wall time budget is checked after steps; validation and final evaluation can extend total runtime."]}
        metrics["timing"] = {"entire_run_seconds": time.perf_counter() - started, "peak_sampled_process_rss_bytes": memory.peak_bytes}
        write_json(output / "metrics.json", metrics)
        write_json(output / "learning-curve.json", {"steps": curve, "validation": validation_history})
        return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/scratch.json"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--corpus", type=Path, default=Path(".cache/scratch-data/input.txt"))
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--max-train-seconds", type=float)
    parser.add_argument("--load", type=Path, help="Generate from an existing run directory without training")
    parser.add_argument("--prompt", default="ROMEO:\n")
    parser.add_argument("--bytes", type=int, default=256)
    args = parser.parse_args()
    if args.load:
        config = json.loads((args.load / "model-config.json").read_text(encoding="utf-8"))
        torch.set_num_threads(args.threads or config["threads"])
        if torch.version.cuda is not None or torch.cuda.is_available():
            raise RuntimeError("This command requires a CPU-only PyTorch build.")
        model = ByteTransformer(config).to("cpu")
        checkpoint = args.load / "model.safetensors"
        expected = json.loads((args.load / "metrics.json").read_text(encoding="utf-8"))["model"]["checkpoint_sha256"]
        if sha256_bytes(checkpoint.read_bytes()) != expected:
            raise ValueError("Checkpoint checksum does not match the training report.")
        model.load_state_dict(load_file(str(checkpoint), device="cpu"))
        assert_cpu(model)
        print(json.dumps(generate(model, args.prompt, args.bytes, config["temperature"], config["top_k"], config["seed"] + 7), ensure_ascii=True))
        return
    if args.output is None:
        parser.error("--output is required when training")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Output directory is nonempty; use a fresh directory.")
    args.output.mkdir(parents=True, exist_ok=True)
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for key in ("steps", "threads", "max_train_seconds"):
        if getattr(args, key) is not None:
            config[key] = getattr(args, key)
    write_json(args.output / "run_status.json", {"status": "running", "started_utc": datetime.now(timezone.utc).isoformat()})
    try:
        metrics = _run(config, args.output, args.corpus, not args.offline)
        write_json(args.output / "run_status.json", {"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(), "steps": metrics["training"]["completed_steps"]})
        print(json.dumps({"status": "complete", "before_byte_perplexity": metrics["before"]["test"]["byte_perplexity"], "after_byte_perplexity": metrics["after"]["test"]["byte_perplexity"], "training_seconds": metrics["training"]["loop_seconds_including_validation"], "peak_rss_bytes": metrics["timing"]["peak_sampled_process_rss_bytes"]}), flush=True)
    except BaseException as exc:
        write_json(args.output / "run_status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed", "error_type": type(exc).__name__, "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
