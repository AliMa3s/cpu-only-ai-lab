# CPU-only AI Lab

Reproducible language-model adaptation and inference using a CPU and system RAM. The project measures whether training improves a held-out task and whether an inference change improves speed without unacceptable task-quality loss.

Read the [project goal and acceptance criteria](GOAL.md).

## What the experiments do

| Experiment | Model | Change being measured |
| --- | --- | --- |
| CPU training | `HuggingFaceTB/SmolLM2-135M-Instruct` | Train LoRA adapters for extracting two integer fields from text into JSON; compare held-out loss and generated answers before and after training. |
| Small-model inference | The same trained 135M model | Merge its saved adapter, compare FP32 with dynamic INT8 linear layers, and retain INT8 only if the validation-quality and measured-speed gates pass. |
| Existing-model inference | Ollama `qwen2.5-coder:7b` | Tune CPU thread count while keeping model digest, quantization, context, and generation budget fixed; confirm the selected setting on separate prompts. |

The training experiment does **not** train the 7B model. The pretrained 135M base was trained elsewhere; this project makes its local adaptation and inference CPU-only. The synthetic task measures copying fields into a specified format, not general reasoning, coding ability, or overall language-model quality.

## Results

Measured on an AMD Ryzen 9 3900X (12 physical cores), approximately 32 GiB RAM, Windows 11. Model workloads ran sequentially. Numerical reports are in [`results/`](results/); downloaded weights, local adapters, and environments remain outside Git.

| CPU training measurement | Before | After |
| --- | ---: | ---: |
| Held-out JSON exact match (32 examples) | 0/32 | 32/32 |
| Held-out target-token loss | 0.541428 | 0.000416 |

The run trained **460,800 LoRA parameters** on a 134,515,008-parameter base for 40 optimizer steps. It processed 160 training examples from the 192-example training split. The optimizer loop took **63.23 seconds including validation** (46.02 seconds of training compute); the full load/train/before-and-after evaluation run took **199.12 seconds**. Sampled peak process RSS was **1.53 GiB**. The saved adapter is approximately **1.86 MB** locally. These are narrow format-and-copying task results, not a general intelligence improvement.

Sources: [training metrics and raw predictions](results/training.json), [learning curve](results/training-curve.json), [resolved training settings](results/training-config.json), [exact synthetic datasets](results/data/), [hardware](results/hardware.json).

The unchanged 7B model's 4-thread candidate showed only **1.013x median paired decoding throughput** versus 12 threads, winning **3 of 6 pairs**. That is not a reliable speedup. The reusable chat command retains the baseline. Aggregate rates were 5.93 versus 5.88 tokens/s; the median paired ratio and aggregate ratio are different summary statistics. [Raw tuning and confirmation results](results/inference.json).

Quantizing all linear layers of the trained small model produced **1.679x throughput** (15.68 to 26.33 generated tokens/s, including prompt processing), but validation JSON accuracy dropped from **32/32 to 19/32**. It was **rejected**. [Full INT8 experiment](results/optimization.json).

Quantizing only MLP projections produced **1.331x throughput**, but validation accuracy was **21/32**, so it was also rejected. Merging the adapter without quantization gave a smaller **1.039x aggregate throughput** improvement (14.64 to 15.21 tokens/s), with **32/32 test answers correct** in both versions. These comparisons have separate paired baselines; their speedup ratios should not be multiplied. [MLP INT8 experiment](results/optimization-mlp.json), [FP32 merging experiment](results/merge.json).

**Selected deployment: merged F16 GGUF in Ollama/llama.cpp, forced to CPU.** The 16-token confirmation measured **3.779x median paired speedup** over merged FP32 PyTorch: aggregate **14.29 -> 54.17 generated tokens/s**, including prompt processing. All **6/6 pairs** were faster; prompt and output token counts matched and sampled generated outputs were identical. Both deployments scored **32/32** on the saved held-out extraction task. This combines a runtime change with FP32-to-F16 storage, not a claim about kernels alone. [Confirmation and raw outputs](results/backend-f16-confirmation.json).

The initial 32-token deployment experiment measured **4.483x median speedup** on three matched pairs. Three other pairs were excluded because Ollama naturally stopped at 18 tokens while PyTorch was forced to 32. The 16-token confirmation above includes both prompts without exclusions and is the primary reported deployment result. [Initial deployment experiment](results/backend-f16.json).

The test set is small and synthetic. These findings establish local CPU training and a faster deployment for this checkpoint/workload, not preserved quality across general language-model tasks. The optimized model is installed locally as `cpu-ai-lab-smollm2:trained-f16`; its weights are not pushed to GitHub.

## Windows setup

Requirements: Git, PowerShell, [uv](https://docs.astral.sh/uv/getting-started/installation/), and sufficient free disk space and RAM for the chosen model. Setup uses Python 3.12 by default. Network access is needed for dependencies and the first model download.

```powershell
git clone https://github.com/AliMa3s/cpu-only-ai-lab.git
Set-Location cpu-only-ai-lab
& .\scripts\setup.ps1

# Set these again in each new shell.
$env:HF_HOME = Join-Path (Get-Location) '.cache\huggingface'
$env:PYTHONNOUSERSITE = '1'
```

The setup script creates `.venv`, installs PyTorch from the CPU wheel index, installs this project in editable mode, and verifies that PyTorch has no CUDA build. It does not install a GPU toolkit or change your installed Ollama models.

The experiment stack pins PyTorch `2.8.0` (CPU wheel), Transformers `4.51.3`, PEFT `0.15.2`, Accelerate `1.6.0`, NumPy `2.2.6`, and psutil `7.0.0`. [`requirements-lock.txt`](requirements-lock.txt) records the installed dependency versions used for the recorded experiments; [`pyproject.toml`](pyproject.toml) pins the direct dependencies.

The base-model revision is pinned to `12fd25f77366fa6b3b4b768ec3050bf629380bac`. Its weights are downloaded from the model publisher and licensed under Apache 2.0; see the [model card](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct).

## Train on CPU

```powershell
& .\.venv\Scripts\python.exe -m cpu_ai_lab.training `
  --config configs/training.json `
  --output artifacts/train-run
```

Use a new output directory for each run: existing nonempty directories are rejected. The first run downloads the pinned base model. Later runs can add `--local-files-only` once it is cached. Optional overrides include `--steps`, `--threads`, `--generation-samples`, and `--save-merged`.

The default configuration performs 40 optimizer steps with six CPU threads, FP32 arithmetic, and rank-eight LoRA updates to attention query/value projections. It uses 192 training, 32 validation, and 32 test records. User and ticket IDs are disjoint across these splits. Prompt tokens and padding are excluded from the training objective; response tokens and the end-of-sequence token are included.

The checkpoint with the lowest **validation** target-token loss is selected, including the untrained baseline as a candidate. Test results do not select checkpoints. Held-out loss is weighted by the number of unmasked, shifted target tokens. Generated answers are scored for exact text, strict JSON validity, and typed JSON equality; extra commentary, duplicate keys, nonstandard numeric constants, and incorrect value types do not count as valid exact answers.

The run writes:

- `adapter/` and `tokenizer/`: reusable local artifacts.
- `data/{train,validation,test}.jsonl`: the exact records used.
- `resolved_config.json`: effective settings, including command-line overrides.
- `baseline.json`, `training_curve.json`, and `metrics.json`: losses, raw generated answers, correctness, timings, sampled peak process RAM, parameter changes, dataset checksums, and software/device evidence.
- `run_status.json`: running, complete, failed, or interrupted status.

Weights and optimizer computation stay on CPU. The default configuration requires a CPU-only PyTorch build, and tensor-device assertions reject non-CPU parameters or buffers. Model loading disables remote Python code.

## Optimize and use the trained model

Run this **after training**, without another training or inference job competing for CPU resources:

```powershell
& .\.venv\Scripts\python.exe -m cpu_ai_lab.optimize `
  --training-dir artifacts/train-run `
  --output results/my-optimization.json `
  --quality-examples 32 `
  --threads 4
```

This reloads the saved adapter, merges it into the pinned base model, and compares merged FP32 with dynamic INT8 linear layers. It uses paired fixed-token generation timings with warmups and randomized variant order. These generation timings **include prompt processing**. Validation quality and measured speed select the variant before final test evaluation; the command can keep FP32 when INT8 is slower or degrades quality.

The default all-linear experiment uses a relative validation-loss tolerance of 10% and requires no reduction in JSON exact match or valid JSON. The follow-up experiment preserves attention and the output head at FP32 and quantizes only MLP projections:

```powershell
& .\.venv\Scripts\python.exe -m cpu_ai_lab.optimize `
  --training-dir artifacts/train-run `
  --output results/my-optimization-mlp.json `
  --threads 4 --quality-examples 32 `
  --quantization-scope mlp-only --max-loss-increase 0.01
```

This follow-up declares an absolute loss allowance because the trained baseline's loss is near zero: candidate loss must be no greater than `max(baseline * 1.10, baseline + 0.01)`. JSON exact match and validity still cannot decrease. This is a different, explicitly recorded validation policy; the original failed all-linear result remains unchanged. Neither policy proves general quality equivalence.

Measure adapter merging alone, without quantization:

```powershell
& .\.venv\Scripts\python.exe -m cpu_ai_lab.merge_benchmark `
  --training-dir artifacts/train-run --output results/my-merge.json --threads 4
```

That comparison uses the same FP32 weights and trained adapter, checks observed logit differences, and evaluates both versions on all held-out test records.

The report includes the decision and raw measurements. A sibling `optimization-recipe.json` records model revision, adapter checksums, versions, CPU settings, and reconstruction steps. It reconstructs the selected variant from the cached base model and local safetensors adapter instead of loading a pickled model object.

```powershell
& .\.venv\Scripts\python.exe -m cpu_ai_lab.optimize `
  --load-recipe results/my-optimization-recipe.json `
  --prompt 'Extract user and ticket as integers. Reply only with JSON containing the keys user and ticket. Record: user=7314; ticket=6289; status=open.' `
  --max-new-tokens 40
```

Keep the referenced adapter directory and cached base model. A JSON recipe alone does not contain weights. The loader checks the recorded package versions and adapter hashes.

## Deploy the trained model to Ollama on CPU

This route merges the adapter in FP32, exports F16 weights, and runs them with Ollama's llama.cpp backend. It changes both runtime and weight-storage precision. The tested Windows Ollama build could not import safetensors directly, so the reproducible path uses the official GGUF converter at a pinned commit.

```powershell
$converter = & .\scripts\get_converter.ps1
& .\.venv\Scripts\python.exe -m cpu_ai_lab.deploy `
  --training-dir artifacts/train-run `
  --output artifacts/deployment-f16 `
  --model cpu-ai-lab-smollm2:trained-f16 `
  --gguf-converter $converter `
  --converter-revision b3fa69955ea81c3b60e77e3c1e2224c4a826ffb9 `
  --create
```

The helper downloads and checksum-verifies the pinned converter source archive. The setup script includes its small tokenizer dependencies. This exact converter/model combination was tested with the pinned training environment; the converter's full requirements for other model architectures are not installed. Export and conversion stay on CPU. The approximately 269 MB GGUF remains local. `--create` imports it into the local Ollama store; existing model names are refused. If an import fails after export, `--resume` verifies and reuses the existing export. No model is uploaded to ollama.com.

Benchmark the deployment with fresh KV state before each timed Ollama call:

```powershell
& .\.venv\Scripts\python.exe -m cpu_ai_lab.backend_benchmark `
  --training-dir artifacts/train-run `
  --deployment-manifest artifacts/deployment-f16/deployment-manifest.json `
  --output results/my-backend-f16.json --threads 4 --tokens 16
```

The harness verifies training/export/model hashes and zero VRAM, compares matching token budgets, records prompt-token counts, and evaluates every saved test example. It unloads and reloads Ollama without a prompt before each timed call, excluding reload time. Both timings include prompt processing; Ollama's wall time also includes HTTP and tokenization. Its first-request kernel initialization can penalize its measured latency. This is an observed deployment comparison, not an isolated kernel benchmark or proof of broad quality equivalence.

Use the trained model:

```powershell
& .\.venv\Scripts\python.exe -m cpu_ai_lab.chat `
  --model cpu-ai-lab-smollm2:trained-f16 --threads 4 --tokens 40 `
  --prompt 'Extract user and ticket as integers. Reply only with JSON containing the keys user and ticket. Record: user=7314; ticket=6289; status=open.'
```

## Tune the existing 7B model

Install and start [Ollama](https://ollama.com/) separately. This benchmark uses an already downloaded `qwen2.5-coder:7b`; it does not download or replace models. If needed, download that model with `ollama pull qwen2.5-coder:7b` before the experiment. Start with Ollama idle and no model loaded.

```powershell
ollama list
ollama ps
& .\.venv\Scripts\python.exe -m cpu_ai_lab.inference `
  --config configs/inference.json `
  --output results/my-inference.json
```

The configuration sweeps 2, 4, 6, 8, and 12 CPU threads against a 12-thread baseline. Every measured run requests `num_gpu=0`, and the harness checks that Ollama reports zero model VRAM allocation and the expected model digest. This is software-reported placement evidence, not independent hardware tracing.

Tuning and confirmation prompts are distinct. The final comparison uses the selected setting on confirmation prompts, with matched baseline measurements. Reports retain per-run responses, timings, context, digest, and thread settings. Ollama's decoding tokens/second excludes loading and prompt processing; the harness also records other latency measures. Do not directly compare that decoding figure with the small-model optimizer's end-to-end generation rate.

The original short proof-of-concept is retained in [`cpu-only-benchmark.ps1`](cpu-only-benchmark.ps1); the Python harness provides the controlled tuning and confirmation workflow.

Use the measured settings for CPU-only generation:

```powershell
& .\.venv\Scripts\python.exe -m cpu_ai_lab.chat `
  --benchmark results/my-inference.json `
  --prompt 'Explain merge sort and write a Python implementation.'
```

The chat command keeps the baseline unless the candidate passes a conservative confirmation check. `--threads` allows an explicit override. It verifies zero model VRAM and leaves the model loaded for two minutes after answering.

## Use your own data

Provide three files in a directory: `train.jsonl`, `validation.jsonl`, and `test.jsonl`. Each line contains nonempty `prompt` and `response` strings; `id` is optional.

```json
{"id":"example-1","prompt":"Extract user and ticket as integers. Reply only with JSON. Record: user=1234; ticket=5678.","response":"{\"user\":1234,\"ticket\":5678}"}
```

```powershell
& .\.venv\Scripts\python.exe -m cpu_ai_lab.training `
  --config configs/training.json `
  --data-dir data/my-task `
  --output artifacts/my-task-run
```

The loader rejects repeated prompts within or across splits. You must also separate related records, sources, and target values appropriately; duplicate-prompt checks cannot detect every form of leakage. Examples exceeding `max_sequence_length` fail instead of being silently truncated. Adjust that configuration for your task and memory budget. For non-JSON tasks, loss and text exact-match remain available, but JSON metrics and the optimizer's JSON quality gate require replacement with task-appropriate evaluation.

## Tests and reproducibility

To export the completed synthetic training experiment for review:

```powershell
& .\.venv\Scripts\python.exe scripts/export_results.py --training-dir artifacts/train-run --output-dir results
```

The exporter includes raw measurements and synthetic data, not weights. It refuses automatic export of custom-data runs because their raw predictions can contain private records.

```powershell
& .\.venv\Scripts\python.exe -m pytest -q
```

Tests cover dataset separation, strict response scoring, token-weighted evaluation, prompt/padding masking, CPU guards, and benchmark selection/measurement behavior. Real model training and inference are explicit commands, not part of the unit-test suite.

To make a comparison useful:

- Run training and benchmarks sequentially. Close unrelated CPU-heavy applications and record relevant system conditions.
- Keep the same model revision or Ollama digest, context, prompt set, token budget, and software versions when comparing settings.
- Use a fresh result/output path for each experiment, preserve raw reports, and compare confirmation results rather than only the best tuning sample.
- Treat seeds and deterministic PyTorch settings as controls, not a guarantee of identical timings or outputs across machines and versions.

Sampled process RSS is not an exact memory peak or whole-machine memory usage. No energy measurement is performed. Short prompt sets and a narrow synthetic task limit generalization. More RAM can fit larger models but does not remove memory-bandwidth or training-compute limits. An optimization that fails to improve speed or quality is reported as such.

## References

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [Hugging Face CPU training documentation](https://huggingface.co/docs/transformers/v5.20.2/cpu)
- [SmolLM2-135M-Instruct model and license](https://huggingface.co/HuggingFaceTB/SmolLM2-135M-Instruct)
- [PyTorch dynamic quantization](https://docs.pytorch.org/docs/stable/generated/torch.ao.quantization.quantize_dynamic.html)
- [Ollama generate API and timing fields](https://docs.ollama.com/api/generate)
- [Ollama running-model placement information](https://docs.ollama.com/api/ps)
- [llama.cpp CPU inference and supported quantization](https://github.com/ggml-org/llama.cpp)
