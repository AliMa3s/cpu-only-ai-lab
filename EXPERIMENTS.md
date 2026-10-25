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

## Longer passages and questions with no supported answer

The next document experiment continues the previously selected 135M document adapter on [SQuAD 2.0](https://rajpurkar.github.io/SQuAD-explorer/), attributed to [Rajpurkar, Jia, and Liang (2018)](https://aclanthology.org/P18-2124/) and released under CC BY-SA 4.0. Its unanswerable questions were written by people to remain relevant to the paragraph while including plausible distractor answers. This experiment uses those native labels; it does not manufacture missing answers by deleting answer spans or cropping the source. Source revision and downloaded-file hashes are recorded and checked.

The fixed subset contains 640 training examples, 128 validation examples, and 256 test examples. Every split is equally divided among four groups: answerable short passages, answerable longer passages, missing-answer short passages, and missing-answer longer passages. Short means at most 192 context tokens; longer means 193 through 512. The complete rendered prompt must fit 608 tokens, every answerable reference at most eight tokens, and the training sequence at most 624. Passages remain complete. These are longer paragraphs than in the first experiment, not long PDFs, complete books, multi-document retrieval, or OCR.

The 400 training articles and 24 validation articles used in the previous document experiment are excluded from the new validation pool. This leaves 18 of the official training source's 442 articles, all reserved for new validation; 12 contain native unanswerable questions. New training uses the remaining article pool. The official development source supplies the local test. All previous 200 test IDs and context hashes are excluded from new test selection, as are cross-split overlaps. The official development articles were nevertheless used in the earlier evaluation, so fresh selected passages do not mean entirely new topics or a previously untouched benchmark. The public provenance and manifest record these exclusions, selected identities, article/context hashes, token lengths, and answerability labels.

Balanced sampling deliberately imposes a 50/50 answerable/missing-answer ratio and equal short/long groups. This is an experimental control, not an estimate of production prevalence or a full SQuAD 2.0 score. Reference-length and context filters also change the source distribution. The warmstart is evaluated again on the same new prompt and subset as the trained candidate; the earlier 52% exact-match result cannot be compared directly with a mixed answerability score here.

Training retains the existing rank-eight q/v adapter and uses a new AdamW optimizer with learning rate `0.0002`. The budget is fixed at 160 optimizer updates, global batch four, with one example from each group per update. This presents all 640 selected training records once. Targets are the first official answer for answerable examples and the exact text `NOT_FOUND` for native missing-answer examples, followed by EOS. Prompt and padding positions remain excluded from response-token loss. The same CPU-only FP32 path and response-position vocabulary projection are used. Training and evaluation run sequentially; the saved configuration records thread and batch settings.

At baseline and every 40 updates, validation computes both response-token loss and generated-answer metrics. A candidate can replace the warmstart only when all predeclared validation checks pass: strict overall exact match and F1 must each improve, answerable exact match and F1 may each decrease by at most 0.05 absolute, explicit correct abstention on missing-answer questions must improve, and answer coverage must remain above zero. Among passing checkpoints, selection maximizes the harmonic mean of answerable exact match and explicit correct-abstention recall, then overall F1, overall exact match, and lower response-token loss. The original adapter remains the fallback if no trained checkpoint qualifies. Test results do not choose a checkpoint or relax these rules.

Strict task scoring recognizes `NOT_FOUND` only after trimming surrounding whitespace, before normalizing ordinary answer text. A blank or malformed answer is tracked separately and does not count as deliberate abstention. For answerable questions, normalized exact match and token F1 use the best official reference. Standard SQuAD-style scores are additionally reported with the marker mapped to an empty answer; the [official scorer](https://github.com/rajpurkar/SQuAD-explorer/blob/master/evaluate-v2.0.py) also normalizes empty, punctuation-only, and article-only outputs to an empty answer, so those scores can differ from the stricter task contract.

Reports separate answerable accuracy, explicit missing-answer recall, false answers on missing-answer questions, false abstentions on answerable questions, malformed outputs, coverage, and answer precision, with short/long breakdowns. An always-abstain policy reaches 50% overall EM/F1 on this balanced sample while answering no answerable question correctly; the harmonic selection metric and retention gates prevent treating that as success. The metrics describe this controlled subset, not calibrated confidence or a guarantee that the model can reliably recognize every unsupported question.

Use the completed first document run as the warmstart. Data preparation checks the previous public split manifest, and training checks that its exclusions correspond to the supplied source adapter's experiment:

```powershell
& .\.venv\Scripts\python.exe -m cpu_ai_lab.document_v2_data `
  --config configs/document-v2.json --download-data `
  --prior-manifest results/document-split-manifest.json `
  --output artifacts/my-document-v2-data

& .\.venv\Scripts\python.exe -m cpu_ai_lab.document_v2 `
  --config configs/document-v2.json --source-run artifacts/document-task-v1 `
  --prepared-data artifacts/my-document-v2-data --output artifacts/my-document-v2-run

& .\.venv\Scripts\python.exe -m cpu_ai_lab.document_v2 `
  --load artifacts/my-document-v2-run --threads 12 `
  --context 'The meeting takes place in Paris on Tuesday.' `
  --question 'What time does the meeting start?'

& .\.venv\Scripts\python.exe -m cpu_ai_lab.document_v2 `
  --verify-reload artifacts/my-document-v2-run `
  --prepared-data artifacts/my-document-v2-data --threads 12

& .\.venv\Scripts\python.exe -m cpu_ai_lab.document_v2 `
  --export-run artifacts/my-document-v2-run --results-directory results `
  --export-prefix my-document-v2
```

The example question intentionally asks for information absent from its passage; it is a functional usage example, not a promised model answer. Use new output directories. If the previous experiment was reproduced under another path, substitute that completed run and its matching split manifest together. Replay checks the first test example in each of the four groups in fixed source order, comparing reloaded serial generation with saved batched output. It does not select examples by correctness or change the chosen checkpoint.

The run preserves both the selected adapter and the final trained adapter locally. Public exports contain configuration, dataset provenance, split manifest, sample schedule, all validation selection trials, training curve, aggregate/per-record metrics, and optional replay evidence. Raw passages, questions, answers, and model weights remain excluded. CPU process RSS is sampled, not an exact peak. Elapsed time includes loading the warmstart, baseline evaluation, training, generated validation, saving, and selected test evaluation; dataset preparation, Python startup/imports, and final report serialization are excluded. Differences in prompt/output lengths prevent interpreting before/after runtime as a matched-work speedup.

### Bounded follow-up: a span-validity guard on the unchanged warmstart

After early LoRA validation checkpoints lost substantial answerable accuracy, a separate deterministic policy was proposed. This is a post-hoc follow-up, not part of the original training protocol and not evidence that the model learned semantic abstention. The primary run, checkpoint rules, and reports remain unchanged. The follow-up reuses only the original warmstart's saved generations, and waits for the primary run to finish before scoring the policy.

The fixed rule strips surrounding whitespace and preserves an exact model-generated `NOT_FOUND`. Otherwise it applies the same lowercase, punctuation/article-removal, and whitespace normalization used for answer scoring. A nonempty normalized output is retained only if it is a literal substring of the normalized complete context; every other output becomes `NOT_FOUND`, explicitly attributed to the guard. The rule accepts only context and model output: gold answers and answerability labels never enter the decision. It has no trained parameters, confidence threshold, or search over alternative rules.

The 128 saved validation generations are checked first. The candidate must pass the identical predeclared validation gate used for LoRA checkpoints, including answerable EM/F1 retention. That decision and a fingerprinted policy recipe are saved before loading the 256 local test generations. Reports then show both the fixed candidate and the selected policy; if validation rejects the candidate, the selected policy remains the unguarded warmstart even if test scores would favor the guard. There is no additional model inference during offline scoring and no inference-speed claim.

```powershell
& .\.venv\Scripts\python.exe -m cpu_ai_lab.document_guard `
  --run artifacts/my-document-v2-run --prepared-data artifacts/my-document-v2-data `
  --output artifacts/my-document-v2-guard

& .\.venv\Scripts\python.exe -m cpu_ai_lab.document_guard `
  --load-policy artifacts/my-document-v2-guard/policy.json `
  --source-run artifacts/document-task-v1 --threads 12 `
  --context 'The meeting takes place in Paris on Tuesday.' `
  --question 'What time does the meeting start?'

& .\.venv\Scripts\python.exe -m cpu_ai_lab.document_guard `
  --verify-reload artifacts/my-document-v2-guard/policy.json `
  --source-run artifacts/document-task-v1 --prepared-data artifacts/my-document-v2-data `
  --threads 12
```

The offline command checks the completed primary report, generation configuration, dataset provenance, split manifest, saved local prediction hashes, identities, and recomputed warmstart scores. `report.json` binds `policy.json` by SHA256. The policy records its source adapter checksums, warmstart report hash, prompt fingerprint, rule definition, and generation configuration. Live loading verifies that source again, runs it on CPU, and distinguishes model-generated abstention from guard-induced abstention. Fixed replay uses the first test record from each of the four groups and separately checks the original model generation and the policy output against saved hashes. Public publication is limited to the policy, report, and optional replay evidence; these contain hashes and scores rather than source passages or generated answer strings.

A guard-induced abstention means **no supported span was extracted**, not that the document necessarily lacks an answer. Wrong distractor spans and word fragments can pass a substring check. Partially correct non-span answers may be rejected. Report guard triggers separately by answerability and passage length, together with answer retention and remaining false answers. Any improvement is attributable to this limited output-validity policy over the existing model, not to improved underlying reading comprehension.

### Audit, paired analysis, and figures for this milestone

`scripts/profile_document_v2_data.py` uses only JSON manifests and saved settings to check the current split, length ranges, and previous training overlap. It reconstructs the earlier selected checkpoint's first 300 examples from its saved shuffle seed; this is explicitly reconstructed exposure rather than a sampler trace. The current training set shares two question IDs and 12 passages with that selected prior exposure. Continued adaptation intentionally reuses training material. New validation and test exclusions are checked independently.

`scripts/analyze_document_v2.py` accepts either the primary metrics or the guard report. It aligns IDs, article/context/question hashes, lengths, and answerability before counting exact-score transitions. It resamples entire test articles jointly for baseline and selected policy, with replacement, using 10,000 seeded draws, then recomputes equal-weight means across the four strata. Draws lacking a stratum are skipped and counted, rather than silently replaced. Percentile 95% intervals describe variation under this article resampling scheme; they do not establish universal performance or select a checkpoint. The primary fallback comparison has identical predictions, so its paired-delta intervals are zero, which does not imply certainty about model accuracy. The separate guard analysis retains its own source-report checksum and comparison label.

```powershell
& .\.venv\Scripts\python.exe scripts/profile_document_v2_data.py `
  --current-manifest artifacts/my-document-v2-data/split_manifest.json `
  --current-provenance artifacts/my-document-v2-data/data_provenance.json `
  --output results/my-document-v2-data-audit.json

& .\.venv\Scripts\python.exe scripts/analyze_document_v2.py `
  --metrics results/document-v2-metrics.json --output results/my-document-v2-analysis.json

& .\.venv\Scripts\python.exe scripts/analyze_document_v2.py `
  --report results/document-v2-guard/report.json --output results/my-document-v2-guard-analysis.json

& .\.venv\Scripts\python.exe scripts/plot_document_v2.py
```

The plotting command uses the published primary metrics, validation trials, and guard report to write `results/figures/document-v2.png` and `.svg`. It requires the existing optional plotting dependencies. Keep the guard's `policy.json` and hash-bound `report.json` together when publishing or moving them; the whitelist for publication is those two files plus `reload_verification.json`, without source passages, generated answer strings, or model weights.

## Learning when to withhold an answer on CPU

This follow-up freezes the original selected document adapter and trains one small logistic correctness verifier. Its training target is 1 only when the raw generator gives an exact-correct answer to an answerable question. Incorrect answers and outputs on missing-answer questions are negative examples. The classifier learns 20 coefficients and an intercept; no language-model weights are updated. A verifier may retain the original answer or replace it with `NOT_FOUND`.

This is a bounded test of selective answering, motivated by [Kamath, Jia and Liang (2020)](https://aclanthology.org/2020.acl-main.503/). The paper motivates a learned error detector; its reported performance is not a result for this model or machine.

### Fresh passages and fixed recipe

`configs/document-verifier.json` fixes seed 20261030, 1,024 training examples, 120 validation examples, and 348 test examples, equally divided among native answerability and short/long passage groups. Every question ID and exact context from **all splits of both earlier document experiments** is excluded from every new split. The current splits are article-disjoint. Validation uses the same 18 official-training articles held out from the original adapter's train/validation pool; test uses official-development articles. Their topics appeared in earlier evaluations, so fresh passages are not an independent corpus replication. Possible pretraining exposure and paraphrases remain uncontrolled.

Initial targets were 256 validation and 512 test examples. Historical exclusions left only 30 distinct missing-long validation contexts and 175 distinct long test contexts. With one question per context and four equally sized groups, the maximum balanced sizes are 120 and 348. These counts were chosen from source availability before model generation. The preparation report records the limits and all exclusions. Complete paragraphs, the unchanged prompt, pinned source bytes, and token budgets are inherited from the preceding experiment.

Features use only the complete passage, question, raw generated answer, and generation distributions. They include six confidence statistics and 14 fixed lexical features listed in the saved model's feature contract. Confidence is computed from processed greedy-generation scores: mean/minimum chosen-token log probability, mean entropy in nats, mean top-1 minus top-2 probability, generated token count, and EOS termination. Confidence averages exclude EOS and later batch padding; an EOS-only generation uses its one EOS distribution. Token count includes EOS. The generator uses FP32, 12 CPU threads, deterministic greedy decoding, length-grouped batches of eight, and a 16-token cap.

Only training features fit the mean/standard-deviation scaler. A deterministic float64 Newton solver minimizes mean binary log loss plus `0.5 * 0.01 * sum(coefficients**2)`; the intercept is unpenalized. The solver has at most 100 updates, fixed damping and line search, and saved convergence evidence. There is one feature recipe, one regularization setting, and one fit. Scores are not calibrated probabilities for real user queries; this experiment's artificial answerability mixture differs from deployment prevalence.

### Validation selection and failure behavior

Thresholds are the fixed grid 0.00 through 1.00 in increments of 0.01, plus a named always-abstain diagnostic. A score strictly below the threshold suppresses the answer. Existing model abstentions remain abstentions. A deployable verifier must satisfy all three validation requirements:

- Retain at least 95% of the raw generator's exact-correct **answerable** responses; a zero baseline-correct denominator fails.
- Withhold at least 40% of the missing-answer cases.
- Lose at most 0.02 absolute answerable token F1.

Among passing thresholds, rank by missing-answer recall, then answerable F1, answerable exact match, coverage, and earlier grid entry. If none passes, select unchanged raw model output. Still publish the best threshold satisfying the retention/F1 constraints as a **diagnostic candidate**, even if it misses the recall target. Test results do not promote that candidate. A raw-confidence comparator uses `exp(mean_logprob)` and the same validation search; the fixed span guard is also evaluated without tuning it.

The model, validation selection, and deployment policy are serialized before **any test generation**. The report records timestamps and hashes. Implementation fingerprints use canonical LF newlines so Git newline conversion does not change them. Loading verifies the feature contract, model integrity, policy siblings, original adapter source, prompt, and implementation version. The frozen adapter state hash is checked before and after the experiment. Public metrics distinguish deployed fallback from a rejected candidate and retain strict abstention, answer quality, coverage, and per-group results.

### Reproduction and complete request costs

Use the existing CPU-only environment, cached original document adapter, and pinned SQuAD2 files in `.cache/document-v2-data`. On first preparation, add `--download-data` to fetch the checksum-verified dataset files. If reproducing the earlier experiments with different saved manifests, supply both via `--prior-manifests PATH_TO_ORIGINAL_MANIFEST PATH_TO_V2_MANIFEST`, in that order; the first must match the frozen generator's source run. Choose new output directories when repeating a run; previous artifacts are preserved.

```powershell
$env:HF_HOME = Join-Path (Get-Location) '.cache\huggingface'
$env:HF_HUB_OFFLINE = '1'
$env:PYTHONNOUSERSITE = '1'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'

& .\.venv\Scripts\python.exe -m cpu_ai_lab.document_verifier_data `
  --config configs/document-verifier.json --output artifacts/my-verifier-data

& .\.venv\Scripts\python.exe -m cpu_ai_lab.document_verifier_run `
  --config configs/document-verifier.json --prepared-data artifacts/my-verifier-data `
  --source-run artifacts/document-task-v1 --output artifacts/my-verifier-run

& .\.venv\Scripts\python.exe -m cpu_ai_lab.document_verifier_run `
  --benchmark-run artifacts/my-verifier-run --prepared-data artifacts/my-verifier-data `
  --benchmark-mode paired --output artifacts/my-verifier-run/benchmark-paired.json

& .\.venv\Scripts\python.exe -m cpu_ai_lab.document_verifier_run `
  --benchmark-run artifacts/my-verifier-run --prepared-data artifacts/my-verifier-data `
  --benchmark-mode baseline --output artifacts/my-verifier-run/benchmark-baseline.json

& .\.venv\Scripts\python.exe -m cpu_ai_lab.document_verifier_run `
  --benchmark-run artifacts/my-verifier-run --prepared-data artifacts/my-verifier-data `
  --benchmark-mode verifier --output artifacts/my-verifier-run/benchmark-verifier.json

& .\.venv\Scripts\python.exe scripts/export_document_verifier.py `
  --run artifacts/my-verifier-run --output results/my-verifier

& .\.venv\Scripts\python.exe -m cpu_ai_lab.document_verifier_run `
  --load-policy results/my-verifier/policy.json --source-run artifacts/document-task-v1 `
  --context 'The laboratory opened in Paris in 1999.' --question 'When did the laboratory open?'
```

The full experiment timer includes model loading, all training/validation/test generation and feature extraction, classifier fitting, and selection. It excludes interpreter startup, dataset preparation, and final report serialization. The classifier-fit timer alone is **not** the training pipeline's total cost.

The paired request benchmark uses the first three test examples in each stratum, fixed by source order, repeated three times. Baseline and verifier arm order alternates. Warmups and model loading are reported separately. The baseline does not collect distributions or features. The verifier arm includes generation, confidence extraction, lexical features, scoring and application of the diagnostic threshold. Every pair records original output hashes and token work; unmatched pairs remain visible and do not enter the paired latency ratio. Separate fresh-process single-arm probes report full sampled process RSS so an allocator shared by the paired run cannot be misreported as incremental verifier RAM. Run model jobs sequentially.

The cost benchmark deliberately measures the candidate even if its quality gate failed. Its overhead then describes a rejected experiment, not the selected deployment. Replay evidence compares the serial request outputs and classifier scores with the saved batched evaluation. Publication exports only whitelisted numerical artifacts, including the small learned classifier coefficients; passages, generated answer strings and language-model weights remain local. Keep `policy.json`, `verifier.json`, `selection.json`, and `protocol.json` together.

### Independent verification and descriptive uncertainty

`scripts/audit_document_verifier.py` independently reconstructs the gold-free lexical features, training labels and scaler, refits the small classifier, replays every validation threshold, recomputes held-out metrics, checks full native passages and historical exclusions, and verifies hashes and publication privacy. It does not repeat language-model inference or independently reconstruct confidence from token logits; fixed live replay and mocked confidence tests cover those separate concerns.

`scripts/analyze_document_verifier.py` jointly resamples the 35 held-out article clusters 10,000 times with seed 20261031. It gives the four strata equal weight and compares the diagnostic candidate and actual selected policy with the raw baseline. It reports exact-correct retention as a level and other scores as paired changes. Empty-stratum draws and zero-denominator retention draws are counted explicitly. These post-selection intervals are descriptive, and do not alter validation gates. An unchanged fallback has zero paired differences by construction, not zero uncertainty about its absolute accuracy.

```powershell
& .\.venv\Scripts\python.exe scripts/analyze_document_verifier.py `
  --report results/my-verifier/report.json --output results/my-verifier/analysis.json

& .\.venv\Scripts\python.exe scripts/audit_document_verifier.py `
  --run artifacts/my-verifier-run --prepared artifacts/my-verifier-data `
  --public results/my-verifier --output results/my-verifier/audit.json

& .\.venv\Scripts\python.exe scripts/plot_document_verifier.py `
  --report results/my-verifier/report.json --selection results/my-verifier/selection.json `
  --benchmark results/my-verifier/benchmark-paired.json --output results/figures/my-verifier
```

For the recorded run, threshold 0.11 achieved the two headline validation targets (40% missing recall, 95% exact-correct retention), but lost 5.25 points of answerable F1. The added two-point partial-answer quality limit therefore rejected it. The best candidate within both quality constraints used threshold 0.08 and reached only 30% validation missing recall. That diagnostic threshold remained fixed on test; no additional threshold was selected from test outcomes.
