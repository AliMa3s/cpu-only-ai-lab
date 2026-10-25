# Follow-up experiment methods and commands

These experiments extend the initial synthetic extraction proof. They run sequentially on CPU; do not launch training alongside a timing benchmark. Results describe this machine and these workloads, not general performance or quality guarantees.

Use the environment created by `scripts/setup.ps1` and set the local cache in each shell:

```powershell
$env:HF_HOME = Join-Path (Get-Location) '.cache\huggingface'
$env:PYTHONNOUSERSITE = '1'
```

## Longer inference

First train and deploy the extraction adapter using the README commands. This experiment checks the saved adapter, export hashes, installed Ollama digest, and zero VRAM allocation.

```powershell
& .\.venv\Scripts\python.exe -m cpu_ai_lab.extended_benchmark `
  --config configs/extended-benchmark.json `
  --output results/my-extended-benchmark.json
```

The FP32 PyTorch baseline uses SDPA, a merged adapter, and KV caching. It selects one of 1, 2, 4, 6, or 12 threads on two separate tuning prompts with two repetitions. Selection requires identical generated work across the settings. Ollama uses the previously chosen four threads and the F16 GGUF of the same merged checkpoint. Both runtime and storage precision change.

Confirmation uses four predetermined raw-completion prompt families, 128- and 256-token budgets, and three repetitions: 24 pairs with randomized pair order and balanced backend order. Both backends stop naturally. Every request is retained; the fixed-work speed summary requires both responses to reach the budget, equal prompt token counts, and no reported cached prompt tokens. Early stops are reported separately. Repetitions are repeated observations of the same four prompts, not 24 independent tasks.

Each Ollama call explicitly unloads the model, verifies absence, and reloads it without a prompt before timing. Reload time is excluded; first-call kernel overhead remains included. PyTorch starts a fresh generation cache. Wall time includes tokenization, generation, and response decoding; Ollama also includes local HTTP/stream overhead. Time to first content means first nonempty decoded content at the client, and Ollama may deliver multiple tokens per chunk. PyTorch forward-hook durations and Ollama server prefill/decode durations use different boundaries and are not directly comparable.

RAM sampling records the Python process and the Ollama process tree separately. Both stacks can remain resident, and shared pages can be counted more than once. These are sampled RSS observations, not isolated model requirements or exact peaks. No power/energy measurement is performed. The original recorded benchmark's name-based sampler missed `llama-server.exe`; its Ollama RAM totals are incomplete. [The correction note](results/extended-benchmark-notes.json) preserves that report and explains the corrected sampling method. A separate memory-only follow-up captures the complete process tree without a resident PyTorch model:

```powershell
& .\.venv\Scripts\python.exe -m cpu_ai_lab.memory_probe --output results/my-backend-memory.json
```

Raw-completion prompts exercise throughput. They are not a benchmark of useful prose, correct code, or the trained JSON task. A model can generate the same wrong answer faster.

## Real-data classification, small and larger models

The dataset is the [UCI SMS Spam Collection](https://archive.ics.uci.edu/dataset/228/sms+spam+collection), attributed to Almeida and Hidalgo (2011), [DOI:10.24432/C5CC84](https://doi.org/10.24432/C5CC84), licensed by UCI under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The archive SHA256 is pinned in the code. Raw messages remain in ignored local artifacts; public reports contain source row IDs, hashes, labels, and predictions.

Before splitting, normalization groups case/punctuation/digit/URL/email variants. Longer templates are additionally grouped using word-trigram Jaccard similarity of at least 0.8, with transitive components; conflicting-label groups are removed. One representative per group remains. This reduces repeated templates but does not guarantee distinct senders or sources, and the historical dataset may have appeared in the pretrained model's original data.

Both models use identical seeded splits: 1,400 available training records, 192 validation records, and 512 held-out test records. Both receive 120 optimizer steps with balanced four-example batches. That is a fixed update/example budget, not an equal compute budget or a full pass through the available training records. Reports record unique rows seen, including those seen by the selected checkpoint. The test set never chooses a checkpoint.

The language model is used as a **constrained binary classifier**: only the logits for the single-token labels `0` and `1` enter the decision and cross-entropy objective. Scores are conditional on these labels and are not calibrated production spam probabilities. Long message content is explicitly truncated to fit 192 total input tokens; truncation counts are reported. Validation macro F1 selects the checkpoint, with validation loss as a tie-breaker and the initial model included as a candidate.

Baselines include the training-majority label, multinomial Naive Bayes using all available training rows, and another Naive Bayes model using only the unique rows seen by the selected LoRA checkpoint. Naive Bayes smoothing is selected on validation data. The all-data baseline and LoRA do not have equal data exposure; the matched subset is included to make that distinction visible.

Download both pinned checkpoints once (omit any download already cached). No model code is downloaded or executed:

```powershell
& .\.venv\Scripts\python.exe scripts/cache_model.py --config configs/real-task.json
& .\.venv\Scripts\python.exe scripts/cache_model.py --config configs/real-task-360m.json

& .\.venv\Scripts\python.exe -m cpu_ai_lab.real_task `
  --config configs/real-task.json --output artifacts/my-sms-135m --download-data
& .\.venv\Scripts\python.exe -m cpu_ai_lab.real_task `
  --config configs/real-task-360m.json --output artifacts/my-sms-360m
```

`--download-data` only downloads the checksum-pinned public dataset if missing. Models are loaded from the local cache with remote Python code disabled. `--steps` and `--threads` are optional overrides. Each run needs a new output directory.

Use a saved classifier without retraining:

```powershell
& .\.venv\Scripts\python.exe -m cpu_ai_lab.real_task `
  --adapter-run artifacts/my-sms-135m `
  --predict 'Can we meet at the station at six?'

& .\.venv\Scripts\python.exe -m cpu_ai_lab.nb_predict `
  --run artifacts/my-sms-135m `
  --text 'Can we meet at the station at six?'
```

The standalone `nb_predict` command uses only Python's standard library. It verifies the saved recipe checksum and avoids loading PyTorch, Transformers, or a pretrained model. Its log-scores are decision scores, not calibrated probabilities.

This is a research classifier evaluated on a historical dataset, not a measurement of current production spam-filter performance.

### Validation-selected decision thresholds

The original classifier uses a fixed 0.5 threshold. A separate follow-up uses the declared grid 0.05, 0.10, ..., 0.95 and selects the highest validation macro F1, breaking ties by distance to 0.5 and then the lower threshold. BEFORE and AFTER checkpoints each receive their own validation-selected threshold; only then are their test records scored. This adjusts the decision rule, not the probability calibration. It was proposed after early validation behavior showed excessive false positives and before inspection of the trained-model test result. The initial pretrained test summary had already been seen; this is a follow-up analysis, not a fully blinded preregistered study. Original fixed-threshold reports remain primary and unchanged.

```powershell
& .\.venv\Scripts\python.exe -m cpu_ai_lab.calibrate_classifier `
  --training-dir artifacts/my-sms-135m --output results/my-sms-135m-threshold.json

& .\.venv\Scripts\python.exe -m cpu_ai_lab.calibrate_classifier `
  --training-dir artifacts/my-sms-135m --report results/my-sms-135m-threshold.json `
  --predict 'Can we meet at the station at six?'
```

## A model trained entirely from random weights

`scratch.py` creates a 1,902,720-parameter causal transformer from random initialization. It loads no pretrained weights or learned tokenizer. Its fixed 256-symbol vocabulary represents UTF-8 bytes. Four transformer layers use causal SDPA, learned position embeddings, and FP32 AdamW on CPU.

The corpus is the tiny Shakespeare subset in [karpathy/char-rnn at a pinned revision](https://github.com/karpathy/char-rnn/tree/6f9487a6fe5b420b7ca9afb0d7c078e37c1d1b4e/data/tinyshakespeare). Its checksum is checked before use. The upstream README declares MIT; the underlying Shakespeare works are public domain. Only provenance and generated samples are published here, not the corpus or upstream code.

The first 90% of source bytes are training, the next 5% validation, and the rest test. Sampling never crosses these boundaries. The fixed byte vocabulary is not learned from held-out data. Natural repeated phrases can still occur across the text. Evaluation scores every next-byte target exactly once, resetting context every 128 bytes and masking the final partial block.

```powershell
& .\.venv\Scripts\python.exe -m cpu_ai_lab.scratch `
  --config configs/scratch.json --output artifacts/my-scratch-run

# Later runs can add --offline once the corpus is cached.
& .\.venv\Scripts\python.exe -m cpu_ai_lab.scratch `
  --load artifacts/my-scratch-run --prompt 'ROMEO:' --bytes 256
```

The configuration requests 1,200 steps with a 360-second training-loop budget, checked after each step. Validation and final evaluation can extend total elapsed time. The best full-validation checkpoint is saved as safetensors and reloaded with an exact state checksum check. Random-model and train-only unigram losses are included as baselines. A 20-step run was performed to check runtime before the full run; it did not change the model or data settings.

Byte loss/perplexity cannot be compared directly to subword-token loss/perplexity from the other models. Falling byte loss demonstrates learning structure in this small corpus; it does not establish a useful general assistant.

## Recreate the figure

After exporting the numerical reports to `results/`, the optional plotting dependency can render PNG and SVG figures:

```powershell
uv pip install --python .venv/Scripts/python.exe -r requirements-plots-lock.txt
& .\.venv\Scripts\python.exe scripts/plot_followup.py
```

Keep model weights, raw SMS messages, trained Naive Bayes vocabulary files, and environment/cache files out of Git. The existing synthetic-data exporter is specific to the first experiment and should not be used for these runs.
