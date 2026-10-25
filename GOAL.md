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

The practical objective is the best useful model within a local time and RAM budget. A smaller classical model may be preferable to a language model for a narrow task. Higher generated-token throughput alone does not establish better answers.

## Completed follow-up milestones

- Longer inference: 18 matched 128/256-token pairs, 4.767x median paired speedup; six early-stop pairs retained separately. See the [raw report](results/extended-benchmark.json) and [memory correction](results/extended-benchmark-notes.json).
- Real data: CPU LoRA on 135M and 360M bases, shared 512-message test set, classical and matched-data baselines, validation-only checkpoint and threshold decisions. The larger model did not improve measured accuracy at the tested budget. See [methods](EXPERIMENTS.md).
- Training from scratch: 1.90M random-initialized parameters, 1,200 CPU updates, held-out byte perplexity 7.04 versus a 28.85 unigram baseline, reusable checksum-verified checkpoint. See [results](results/scratch.json).

These milestones are bounded local experiments. They do not establish general intelligence, frontier-scale CPU training feasibility, or universal speedups.
