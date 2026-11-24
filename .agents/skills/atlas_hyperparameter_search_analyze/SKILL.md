---
name: atlas_hyperparameter_search_analyze
description: Summarize and analyze a completed hyperparameter search and write the final study summary.
---

# Summarize a completed study

Use the objective, knowledge pack, trial history (`trials.csv`), and structured leaderboard/effect data supplied by the caller. The result is the study's `summary.md`: the one document someone reads to learn what the study did and found.

Write these sections, in this order:

```
# Atlas Summary

## Objective
<the objective in a few lines>

## Counts
- Trials: <total>
- Accepted: <number with status accepted>
- Rejected: <number with status rejected>
- Failed: <number with status failed or timeout>
- Early stopped: <number with status early_stopped>

## Best Trial
- Trial: <trial_id>
- Score: <score> (<score_name>)
- Decision: <decision>
- Configuration: <the config patch>

## Analysis
<outcomes overall and useful trends across the tested configuration axes>

## Next Suggestions
<the open suggestions from the trials, as "- <trial_id>: <suggestion>", and anything the results now point to>
```

Rules:

- Take every count, score, trial identifier and configuration value from the supplied data; count the statuses from the trial history rather than estimating them. If no trial has been accepted yet, say so under Best Trial.
- Distinguish observations from explanations; do not claim causation from insufficient comparisons.
- Mention failed and early-stopped trials when they inform what should not be repeated.
- Return `status: terminal`, a short `message`, and the complete summary in `summary_markdown`; do not modify study state or other files.
