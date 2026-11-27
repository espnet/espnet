---
name: atlas_hyperparameter_search_research_context
description: Write the first version of a hyperparameter search study's knowledge pack.
---

# Start the knowledge pack

The knowledge pack is the study's running notebook. Every proposal, debugging and reflection step reads it, and one entry per finished trial is appended to it later. You write its first version, once, before the first proposal.

Read these files, using the paths in the supplied context (`study_dir`, `recipe_dir`):

- `<study_dir>/program.md`: the objective.
- `<study_dir>/atlas.yaml`: the study configuration, including the recipe's training, inference and metrics config paths, the metric, the budget and the search space.
- `<recipe_dir>/src-checklist.md`: a persistent research checklist. It may not exist.
- `<study_dir>/trials.csv`, only if it exists: trials already recorded when the study was resumed.

Return `status: success`, a one-line `message`, and `pack`: the pack as Markdown. Do not write any file yourself.

## What the pack contains

```
# Knowledge Pack

## Objective
<the objective, as written in program.md, tightened only where it is repetitive>

## Setup
- Recipe dir: <recipe_dir>
- Study dir: <study_dir>
- Training config: <path>
- Inference config: <path>
- Metrics config: <path>
- Metric: <name> (<min or max>)

## Persistent Research Checklist
Checklist path: <path>
<the checklist's items, condensed; "(checklist not found)" if there is none>

## Prior Trials
<one line per trial in trials.csv: id, status, score, decision; "- none" if there are none>
```

## Rules

- Take every path, value and trial result from the files; never invent one.
- Keep it short and factual. Later entries add the lessons; this is only the starting point.
- Do not propose experiments and do not edit configs, source files or study state.
