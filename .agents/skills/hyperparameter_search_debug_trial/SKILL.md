---
name: hyperparameter_search_debug_trial
description: Diagnose a failed hyperparameter search trial and propose a minimal retry fix.
---

# Debug a trial

Inspect the supplied failure logs, resolved training configuration, base configurations, and trial history.

- Identify the root cause from evidence in the logs; do not guess when evidence is missing.
- Make only changes needed to fix the failure. Avoid unrelated hyperparameter changes.
- Return dotted training config patches when a config-only fix is appropriate. The caller validates allowed keys.
- Source edits must follow the allowed directory and tool constraints in the request.
- Set `should_retry` only when a concrete fix can be tried; otherwise explain why the trial should continue without retry.
- Return exactly the structured fields requested by the caller.

## Common failure guidance

- **CUDA OOM:** reduce `dataloader.train.iter_factory.batches.batch_bins`, initially by about half. Typical single A100 40 GB starting points are 1,500,000 for 80-dim filterbanks, 400,000–800,000 for raw waveform models, and 200,000–400,000 for large SSL models. Increasing `trainer.accumulate_grad_batches` can preserve effective batch size.
- **NaN/Inf loss:** consider lowering `optimizer.lr`, adding or reducing `scheduler.warmup_steps`, reducing `trainer.gradient_clip_val`, or using a numerically stable precision such as `bf16-mixed`.
- **Immediate config/import crash:** use the traceback and resolved config to correct the specific invalid key, target, or missing value.
- **Slow convergence without a hard failure:** do not retry solely for slow progress; allow the trial to complete unless the loss clearly diverges.

Only touch the source paths and tools allowed by the request. A config retry should only include keys that address the observed failure.
