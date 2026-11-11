---
name: hyperparameter_search_propose_trial
description: Propose a new hyperparameter search trial using the study context and constraints.
---

# Propose a trial

Use the objective, knowledge pack, trial history, base configurations, and inflight patches supplied in the request.

- Propose only training configuration changes allowed by the request.
- Keep `trainer.max_epochs`, every `dataloader.*` setting, `dataset.train`, and
  `dataset.valid` fixed across trials.
- Use dotted config keys and do not include runtime paths such as `exp_dir`, `recipe_dir`, `data_dir`, or `exp_tag`.
- Do not repeat a prior or currently inflight patch, or propose settings that
  resolve to a training config already tried. Prefer a small, interpretable
  change that tests one hypothesis.
- Keep inference and metrics configurations unchanged.
- State the rationale, expected effect, risks, and any required extra stages.
- Return exactly the structured fields requested by the caller. Do not edit files.
