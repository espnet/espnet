---
name: hyperparameter_search_analyze
description: Analyze completed hyperparameter search results and write the final study report.
---

# Analyze a completed study

Use the objective, knowledge pack, trial history, and structured leaderboard/effect data supplied by the caller.

- Summarize the best result, overall outcomes, and useful trends across tested configuration axes.
- Distinguish observations from explanations; do not claim causation from insufficient comparisons.
- Mention failed and early-stopped trials when they inform what should not be repeated.
- Keep all scores, trial identifiers, and configuration values consistent with the supplied data.
- Return `status: terminal`, a short `message`, and the complete report in `analysis_markdown`; do not modify study state or other files.
