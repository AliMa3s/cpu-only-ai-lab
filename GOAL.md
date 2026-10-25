# Project goal

Make local language-model training, adaptation, and inference reproducible on an ordinary CPU and system RAM, without GPU computation.

## Acceptance criteria

1. Train real model parameters on the local CPU and save a reusable adapter.
2. Report pre/post held-out target-token loss, generated-task correctness, wall-clock training time, sampled peak process RAM, and CPU-only execution evidence.
3. Improve inference where measurements support it. Tune the same installed 7B model with identical model weights, quantization, context, and token budgets, then confirm the chosen setting on separate prompts.
4. Independently benchmark the trained small model with FP32 versus dynamic INT8 linear layers and report quality changes as well as speed.
5. Publish source, pinned experiment configurations, tests, and raw numerical results. Keep downloaded models, environment files, and private machine paths out of Git.

## Scope and interpretation

The initial training experiment adapts SmolLM2-135M-Instruct to a synthetic record-to-JSON extraction task. It is not training the installed 7B model, pretraining from scratch, or evidence of general intelligence improvement. The 7B Ollama speed experiment is separate and uses unchanged model weights.

The base model was pretrained elsewhere; this project makes the local adaptation and inference stages CPU-only. All results are specific to the measured machine and workloads. A failed optimization is a valid result and must be reported.

## Follow-on research

- Validate longer inference with 128- and 256-token budgets, varied predetermined prompts, separate thread tuning, repeated paired timings, first-content latency, and process RAM. Retain early stops and unmatched work in the report.
- Evaluate 135M and 360M models on the same public real-data classification task with at least 500 held-out examples, near-duplicate controls, validation-only selection, and inexpensive classical baselines. Measure useful accuracy as well as model size and CPU cost.
- Train a small causal transformer from random weights, without a pretrained tokenizer or checkpoint, under an explicit CPU time budget. Compare held-out next-byte loss with random initialization and a train-only unigram baseline; save and reload the resulting model.
- Compare optimized CPU backends and lower-bit model formats with task-quality checks.
- Investigate speculative decoding and forward-only adaptation only after establishing reproducible baselines.
- Measure energy per useful answer when suitable hardware telemetry is available.
- Improve semantic missing-answer recognition while retaining correct answers, then evaluate retrieval over genuinely longer documents. The current 512-token passage experiment and substring guard are limited steps toward that goal.
- Evaluate a compact extractive reader with full-parameter CPU adaptation. A replacement must retain at least 95% of the generator's exact-correct answerable cases, catch at least 40% of native missing answers, and lose at most two absolute points of answerable token F1 on validation. Measure latency and RAM on identical complete inputs, and retain the generator if any quality gate fails.

The practical objective is the best useful model within a local time and RAM budget. A smaller classical model may be preferable to a language model for a narrow task. Higher generated-token throughput alone does not establish better answers.

## Completed follow-up milestones

- Longer inference: 18 matched 128/256-token pairs, 4.767x median paired speedup; six early-stop pairs retained separately. See the [raw report](results/extended-benchmark.json) and [memory correction](results/extended-benchmark-notes.json).
- Real data: CPU LoRA on 135M and 360M bases, shared 512-message test set, classical and matched-data baselines, validation-only checkpoint and threshold decisions. The larger model did not improve measured accuracy at the tested budget. See [methods](EXPERIMENTS.md).
- Training from scratch: 1.90M random-initialized parameters, 1,200 CPU updates, held-out byte perplexity 7.04 versus a 28.85 unigram baseline, reusable checksum-verified checkpoint. See [results](results/scratch.json).
- Complete CPU pipeline: fresh same-work SMS run reduced from 553.49 to 232.42 seconds (**2.381x**), exceeding the 2x target while passing the predeclared validation gate and retaining identical predictions on all 512 test records. The update/sample/token schedules were checked; repeated stage confirmations are separate from the single complete-run pair. See [comparison](results/pipeline-comparison.json).
- Real document extraction: CPU adaptation on complete short SQuAD passages improved generated exact answers from 2/200 to **104/200**, with **58.59% token F1**, 192.71 seconds of training/validation, and a verified reusable adapter. Validation chose step 75 of the fixed 100-update run. A 52% exact score still requires human review; that initial run evaluated only answerable short passages. See [results](results/document-metrics.json).
- Longer passages and missing answers: trained for 160 CPU updates on a balanced native SQuAD2 subset, with new validation articles and excluded previous test passages. Every trained checkpoint failed answer-accuracy retention, so the previous adapter was kept. A separate fixed span guard passed validation and caught **15/128 missing test answers**, retaining **66/128 exact answerable answers**, with a 2.28-point loss in answerable token F1. Most missing answers remain undetected; this milestone establishes the benchmark and a limited guard, not reliable semantic abstention. See [training](results/document-v2-metrics.json) and [guard results](results/document-v2-guard/report.json).
- Learned correctness verifier: fitted a 21-parameter CPU classifier on 1,024 fresh examples while freezing the language model. The best validation candidate within the retention/F1 limits retained all 20 exact-correct answers but caught only **30%** of missing answers, below the predeclared 40% target, so it was rejected. On 348 fresh test questions it caught **49/174 missing answers** and retained **86/87 baseline exact-correct answers**, with a 2.20-point answerable F1 loss. The full pipeline took **469.57 seconds** and **1.49 GiB** peak sampled RSS. Thirty-six matched request pairs showed approximately unchanged latency; no speedup is established. The diagnostic candidate detects more missing answers than the fixed span guard at the cost of lower answerable F1; reliable abstention remains unfinished and the saved policy retains the raw generator. See [report](results/document-verifier/report.json) and [validation decision](results/document-verifier/selection.json).

- Compact extractive reader: adapted all **33.2M parameters** on CPU for **256 updates / 2,048 unique questions** in **29 min 26 sec**, with **4.38 GiB** sampled peak process RSS. Microbatches preserved the global loss objective and enabled completion after an initial full-batch run stopped at 15 updates. At a fixed zero threshold, held-out answerable exact match improved **51.17% → 59.77%** and F1 **61.97% → 68.57%**, while missing recall fell **79.30% → 75.39%**. The diagnostic reader achieved an **8.926x median paired request speed ratio**, but failed the validation replacement gate: even always-answer retained only 41/51 baseline-correct cases. The saved policy therefore retains the generator. This is full-parameter local adaptation of a publisher-pretrained, previously GPU-trained QA model, not CPU pretraining from scratch. See [results](results/reader/report.json), [latency](results/reader/benchmark-paired.json), [resource calibration](results/reader-calibration/selection.json) and [methods](EXPERIMENTS.md#a-compact-extractive-reader-on-fresh-movie-plots).

These milestones are bounded local experiments. They do not establish general intelligence, frontier-scale CPU training feasibility, or universal speedups.
