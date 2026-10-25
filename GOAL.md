# Project goal

Make local language-model adaptation and inference reproducible on an ordinary CPU and system RAM, without GPU computation.

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

- Evaluate additional real datasets and larger models under explicit time and RAM budgets.
- Compare optimized CPU backends and lower-bit model formats with task-quality checks.
- Investigate speculative decoding and forward-only adaptation only after establishing reproducible baselines.
- Measure energy per useful answer when suitable hardware telemetry is available.
