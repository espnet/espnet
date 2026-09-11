---
title: What is a recipe
---

# What is a recipe

In ESPnet3, a recipe is an all-in-one project under `egs3/`.
It contains the configs, Python code, stage runner, and output paths for one
system.

::: important
A recipe is the main working unit for running a baseline, adapting it, fine-tuning,
evaluating, and publishing results.
:::

## How files and stages connect

A recipe is made of configuration files, dataset code, and generated outputs that are connected through execution stages.
Click a file or stage below to explore how data flows through the project.

<FileStageMappingWidget />

Stages run in this order (a system only runs the ones you pass to `--stages`):

```text
create_dataset → train_tokenizer → collect_stats → train → infer → measure
  → pack_model → upload_model → pack_demo → upload_demo
```

`TTSSystem` inserts `create_token_list` and `remove_long_short` before `collect_stats`.

`create_dataset` reads `training_config.dataset` and, for each entry, imports the recipe's
`dataset` module and looks up two classes by name: `Dataset` and `DatasetBuilder`. A recipe's
`dataset/__init__.py` must export exactly those two names (aliased from whatever your real
classes are called) — this is how `BaseSystem.create_dataset()` finds your builder without any
extra registration step.

A recipe is the working project you use to:

- run a baseline
- adapt the baseline to your data
- fine-tune an existing model
- run inference and evaluation
- pack and publish outputs

If you clone a recipe into your own directory, you can keep using that cloned
project for the full workflow.

::: note
The cloned recipe is meant to be edited.
You are expected to change configs, dataset wiring, and model settings inside
that project.
:::


## Clone a recipe and keep working in it

When you run:

```bash
espnet3 clone librispeech/asr --project my_project
cd my_project
```

you get a full recipe project ready for baseline runs.
It is also the place where you can later:

- change configs
- swap datasets
- continue training
- fine-tune from a checkpoint or pretrained model
- run inference and measure results

So the cloned recipe is not temporary.
It becomes your working project.

## Typical flow

1. Clone an existing recipe.
2. Run the baseline once.
3. Read the configs and dataset code.
4. Replace the dataset or model settings you need.
5. Re-run training or fine-tuning in the same project.
6. Run inference and measurement.

## Next steps

<DocCards :cols="3">
  <DocCard
    title="Guides"
    desc="Step-by-step guides with code: dataset, builder, configs, and training"
    icon="tabler:code"
    href="../guides/index.html"
  />
  <DocCard
    title="System and Stages"
    desc="See how `run.py`, stages, and recipe execution are wired together."
    icon="tabler:puzzle"
    href="../stages/index.html"
  />
  <DocCard
    title="Recipe template"
    desc="Start a new recipe from scratch using egs3/TEMPLATE/asr"
    icon="tabler:copy"
    href="https://github.com/espnet/espnet/tree/master/egs3/TEMPLATE/asr"
    :external="true"
  />
</DocCards>
