---
name: hyperparameter_search_reflect
description: Reflect on a completed hyperparameter trial and suggest the next experiment.
---

# Reflect on a trial

Use the objective, knowledge, trial history, and latest metrics provided by the caller.

- Summarize what the result says about the tested hypothesis.
- Note useful lessons and a concrete next suggestion; avoid repeating failed or inflight configurations.
- Keep conclusions proportional to the available evidence.
- Do not edit configs, source files, or study state.
- Return `status` as `continue` or `done`, plus `message`, `reflection`, and `next_suggestion`. Set `status: done` when the supplied `stop_after_trial` context is true; otherwise use `continue`.
