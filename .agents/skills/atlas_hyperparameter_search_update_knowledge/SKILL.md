---
name: atlas_hyperparameter_search_update_knowledge
description: Write the knowledge-pack entry for one finished hyperparameter search trial.
---

# Record a trial in the knowledge pack

The knowledge pack is the study's running notebook. Every later proposal reads it, so each entry has to be short, accurate and useful for choosing the next experiment. The caller appends your entry to the pack; you never edit the pack or any other file.

Use the current trial (`current_trial` in the supplied context), the knowledge pack text, the trial history and the latest metrics provided by the caller.

Return `status: success`, a one-line `message`, and `entry`: one Markdown entry for this trial only. Do not repeat earlier entries and do not restate the whole study.

## Entry for a normal trial

```
## Trial <trial_id>
- **status**: <trial status>
- **score** (<score_name>): <score>
- **config_patch**: `<patch as one line of JSON>`
- **rationale**: <first line of the rationale>
- **expected_effect**: <first line of the expected effect>

**Reflection and comparison evidence:**
<the trial's reflection, condensed to what a later proposal needs>

- **next_suggestion**: <first line of the next suggestion>
```

Leave out a line when the trial has no value for it.

## Entry for an early-stopped trial (`status` is `early_stopped`)

```
## Early Stopping Record: <trial_id>

**Config patch (DO NOT reuse this direction):**
<patch as JSON in a fenced block>

**Original rationale:** <first line of the rationale>

**Early stop reason:**
<why it was stopped, from the reflection>

**Lesson:** This config was stopped early due to poor training trajectory. Do not propose similar configurations.
```

## Rules

- Show the patch without the runtime-only keys `recipe_dir`, `data_dir`, `exp_tag`, `exp_dir` and `stats_dir`.
- Take every number, key and value from the supplied context; never invent a result.
- Compare against the earlier entries only to make the lesson sharper (for example, that a direction was already tried), not to rewrite them.
- Keep the entry free of tool output, logs and commentary.
