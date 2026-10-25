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

## CPU pipeline profiling and controlled execution tuning

`scripts/profile_existing.py` separates the recorded SMS training, periodic validation, before/after evaluation, and residual wall times. It also estimates padded token slots and attention shapes from saved sequence lengths. These are historical timings and arithmetic workload estimates, not new measured speedups. The profiler loads no model weights and performs no forward passes.

```powershell
& .\.venv\Scripts\python.exe scripts/profile_existing.py `
  --output results/my-pipeline-profile.json
```

The optional `--include-training-padding` switch additionally requires the original local runs at `artifacts/real-task-135m` and `artifacts/real-task-360m`. It loads their cached tokenizers, reconstructs the seeded balanced sampler, checks the recorded token total, and hashes the ordered source IDs of every global batch. Padding savings do not translate directly into runtime savings: smaller batches can increase dispatch and matrix-operation overhead.

The execution tuner uses a completed 135M SMS run from the earlier commands. It verifies its adapter and saved training records, then draws 64 tuning records and 64 separate confirmation records from **TRAIN only**. Validation and test records do not choose execution settings. The checkpoint used for these short timing trials is already adapted; its original checkpoint selection used validation, and the trials do not replace a full run from the original pretrained model.

```powershell
& .\.venv\Scripts\python.exe -m cpu_ai_lab.pipeline_speed `
  --config configs/pipeline-speed.json --training-dir artifacts/my-sms-135m `
  --output results/my-pipeline-speed.json
```

The search tries 1, 2, 4, 6, and 12 CPU threads; evaluation batches of 4, 8, 16, and 32; length-sorted evaluation; and training microbatches of 1, 2, or 4. This is a hierarchical search, not every joint combination. Evaluation restores original record order and preserves tokenization, sequence limits, labels, and scoring. Training retains the same ordered four-example global batch and one AdamW update, scales each microbatch mean loss by its fraction of the global batch, and clips accumulated gradients once. Each short trajectory starts from cloned adapter weights, the same Torch seed, and a fresh optimizer. FP32 precision and the model architecture stay fixed.

Evaluation candidates require identical predictions and maximum absolute logit, probability, and per-example loss differences at most `0.0001`. Training candidates require adapter differences at most `0.0001`; confirmation also scores their updated models on unused confirmation records, requiring identical predictions and differences at most `0.002`. The explicit evaluation warmups and numerical reference runs are retained in the report but excluded from speed selection. Tuning uses two training steps; confirmation uses three and repeats each baseline/candidate pair three times in alternating order. A changed setting is recommended only when all confirmation quality checks pass, all three pairs are faster, and median paired speedup exceeds `1.03`. Confirmation can reject a tuned setting and fall back to the baseline; it does not search for another winner.

The microbenchmark excludes thread switching, adapter restoration, and optimizer creation from its timed regions. Its training timer includes collation and work counters, while the original SMS training-compute timer excludes collation. RAM is sampled process RSS in a process that may retain allocations across variants. Reported component ratios and RAM observations need those boundaries; use the complete-run wall time for the final speed claim. PyTorch documents CPU intra-operation thread control in [`set_num_threads`](https://docs.pytorch.org/docs/stable/generated/torch.set_num_threads.html). Each complete-run command below starts a fresh Python process, with configured initial thread counts set before model execution.

## Complete-run comparison and validation acceptance

After selecting the execution recipe, run a fresh baseline and optimized pipeline sequentially with the same cached model/data and semantic configuration:

```powershell
& .\.venv\Scripts\python.exe -m cpu_ai_lab.real_task `
  --config configs/real-task.json --output artifacts/my-pipeline-baseline

& .\.venv\Scripts\python.exe -m cpu_ai_lab.pipeline_run train `
  --config configs/real-task.json --execution-report results/my-pipeline-speed.json `
  --output artifacts/my-pipeline-optimized

& .\.venv\Scripts\python.exe -m cpu_ai_lab.pipeline_run compare `
  --baseline artifacts/my-pipeline-baseline --candidate artifacts/my-pipeline-optimized `
  --output results/my-pipeline-comparison.json
```

The configured workload remains 120 optimizer updates with global batch four, validation every 30 steps, and the same before/after validation and test passes. Checkpoint selection remains validation macro F1, then validation loss, including the initial checkpoint. Comparison checks semantic configuration, split manifest, ordered training-sample schedule, exposure/token counts, every configured update and validation pass, and evaluation record identities/order. The optimized schedule hash comes from the IDs actually sampled; the original baseline schedule is reconstructed from its saved training rows and seed.

The acceptance rule is declared before using the complete-run result: final selected-checkpoint **validation accuracy and macro F1 must not decrease**, and validation binary-label loss may increase by at most **0.01 absolute**. The comparison computes this gate independently of test scores. Test metrics and prediction disagreements are then reported descriptively; they never select settings or relax the gate. The reused historical SMS test set is not a new unseen dataset.

A `2x` complete-run target is met only if this validation gate passes and measured baseline wall time divided by optimized wall time is at least two. Neither a faster isolated stage nor padding arithmetic establishes that result. The reported `elapsed_seconds` starts inside `_train` after imports, initial configuration/status writes, and thread/seed setup: it includes data processing, model loading, training, evaluation, and adapter/tokenizer saving, but excludes Python startup/imports, remaining report hashing, and final metrics/status serialization. One fresh process per variant, baseline first, provides one complete-run observation rather than a repeated randomized estimate. Cached files, other machine activity, and thermal conditions can affect it. Repeated short confirmation measurements remain separate evidence; the original historical timings are also distinct from this fresh comparison.

## Generative extraction from real Wikipedia passages

`document_task.py` adapts the same pinned SmolLM2-135M checkpoint to produce a free text answer from a question and passage. It starts a new rank-eight q/v LoRA adapter. The source is [SQuAD v1.1](https://rajpurkar.github.io/SQuAD-explorer/), attributed to [Rajpurkar, Zhang, Lopyrev, and Liang (2016)](https://arxiv.org/abs/1606.05250) and distributed under CC BY-SA 4.0. Source JSON files are pinned to a repository revision and SHA256 checksums.

The default recipe selects 400 training passages, 64 validation passages, and 200 test passages. Training and validation come from disjoint articles in the official training source; 24 eligible articles form the validation pool. The official development source is reserved as the local test, not the official hidden SQuAD test. Sampling rotates across seeded shuffled articles and keeps at most one question per distinct passage. Exact duplicate contexts across official source sets are removed, and article/question/context overlap across selected splits is rejected.

Eligibility is declared by the configuration: complete passage at most 192 tokens, rendered question/passage prompt at most 256, every reference answer at most five, and training sequence at most 264. Gold answer spans are checked against the source, but never used to crop a passage. Questions with unsuitable reference lengths or empty normalized answers are excluded. These filters intentionally produce a short-passage, short-answer subset; they do not represent full-dataset performance.

Prepare the checksum-verified local data, then train without downloading or selecting new examples:

```powershell
& .\.venv\Scripts\python.exe -m cpu_ai_lab.document_task `
  --config configs/document-task.json --prepare-only --download-data `
  --output artifacts/my-document-data

& .\.venv\Scripts\python.exe -m cpu_ai_lab.document_task `
  --config configs/document-task.json --prepared-data artifacts/my-document-data `
  --training-threads 12 --evaluation-threads 12 --loss-batch-size 8 --generation-batch-size 8 `
  --output artifacts/my-document-run
```

The default budget is 100 updates with batch four and validation every 25 updates. Teacher-forced causal cross-entropy supervises the first reference answer plus EOS; prompt and padding positions are masked. The full vocabulary projection is computed only at hidden-state positions with response targets, preserving that objective for the supported Llama architecture. A targeted test compares the loss and LoRA gradients against the ordinary causal-LM forward path on unequal, right-padded sequences. Evaluation loss weights each batch by its response-token count.

Minimum validation response-token loss selects the checkpoint, with the initial model included. Before/after test generation is greedy, length-sorted, left-padded in batches, and limited to 16 new tokens. Output order is restored and per-row token counts stop at the first EOS. Generated answers receive normalized exact match and token-overlap F1 against all reference answers, taking the best reference score. Normalization lowercases, removes punctuation and English articles, and collapses whitespace. The additional passage-span rate is a case-insensitive substring check. Better validation loss does not guarantee better generated-answer scores.

Thread and batch overrides (`--training-threads`, `--evaluation-threads`, `--loss-batch-size`, and `--generation-batch-size`) are stored in the resolved config. The command above uses 12 threads for both stages and batches of eight for both evaluation paths. These are transferred execution choices, not a measured document-task speedup: that claim requires its own matched baseline. Data preparation can occur outside the training timer when `--prepared-data` is used. Both choices must be disclosed when comparing elapsed times.

Reload the saved adapter for a new question, or verify the first three predetermined test IDs against the saved batched prediction hashes:

```powershell
& .\.venv\Scripts\python.exe -m cpu_ai_lab.document_task `
  --load artifacts/my-document-run --evaluation-threads 12 `
  --context 'The meeting takes place in Paris on Tuesday.' `
  --question 'In which city does the meeting take place?'

& .\.venv\Scripts\python.exe -m cpu_ai_lab.document_task `
  --verify-reload artifacts/my-document-run --prepared-data artifacts/my-document-data `
  --evaluation-threads 12

& .\.venv\Scripts\python.exe -m cpu_ai_lab.document_task `
  --export-run artifacts/my-document-run --results-directory results `
  --export-prefix my-document
```

Loading verifies the saved configuration, provenance/manifest, tokenizer, and adapter checksums; replay additionally verifies the prepared data. Serial/batched output differences are reported, not used to choose a model. The export copies only metrics, resolved config, training curve, provenance, and split manifest, preserving their bytes and refusing conflicting existing reports. Raw passages, questions, references, predictions, and model weights remain in ignored local artifacts.

All source questions here are answerable. This does not evaluate abstention, long documents, OCR, production documents, or unseen factual knowledge. Wikipedia/SQuAD may have appeared in base-model pretraining, and article separation cannot exclude every related fact or near duplicate. Small filtered samples, finite generation caps, and possible serial/batched numerical differences limit interpretation. Report the actual generated-answer results even when training loss improves.

After exporting the recorded reports, render the combined pipeline/document figure with the optional plotting dependencies:

```powershell
& .\.venv\Scripts\python.exe scripts/plot_pipeline.py
```

This reads `results/pipeline-comparison.json` and `results/document-metrics.json`, and writes PNG/SVG outputs in `results/figures/`.
