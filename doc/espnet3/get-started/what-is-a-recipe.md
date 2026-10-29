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
The table shows where each part of the workflow is configured.

| Source | Responsibility | Generated artifacts |
| --- | --- | --- |
| `run.py` | Call `launch(ASRSystem)` under a main guard | Stage logs and config snapshots |
| `dataset/` | Load samples and prepare source/manifests with a builder | Recipe-specific `data/` artifacts |
| `conf/training*.yaml` | Model, dataset, tokenizer, trainer and loader settings | Tokenizer, stats and checkpoints |
| `conf/inference.yaml` | Backend, runner, input/output contract and named test sets | One SCP per inference output field |
| `conf/metrics.yaml` | Metric classes and contract input bindings | `metrics.json` and optional alignments |
| `conf/publication.yaml` / `conf/demo.yaml` | Pack and upload explicit bundles | Packed model or demo directory |

`ASRSystem.stages` declares this order. The launcher executes the requested
list in the supplied order; it does not automatically add prerequisites.
`all` expands the declared list, including uploads:

```text
create_dataset → train_tokenizer → collect_stats → train → infer → measure
  → pack_model → upload_model → pack_demo → upload_demo
```

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


## Work in a source recipe

The reviewed implementation has no `espnet3 clone` command. From a checkout:

```bash
cd egs3/mini_an4/esp2_asr
```

Edit that recipe's configs and dataset wiring, or create a new directory under
`egs3/` following the TEMPLATE layout. A recipe can contain relative paths and
shared TEMPLATE assets; copying an arbitrary directory out of the checkout
does not guarantee a self-contained project. Model publication bundles include
explicitly configured code and rewritten config paths for use elsewhere.

## Typical flow

1. Open an existing source recipe.
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
    desc="Start a new recipe from scratch using egs3/TEMPLATE/esp2_asr"
    icon="tabler:copy"
    href="https://github.com/espnet/espnet/tree/master/egs3/TEMPLATE/esp2_asr"
    :external="true"
  />
</DocCards>
