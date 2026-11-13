# From SpeechBrain

If you come from SpeechBrain, the main shift is where workflow logic lives.

SpeechBrain often puts a lot of behavior into:

- the Brain class
- hyperparameter files
- dataset preparation scripts

ESPnet3 splits that across:

- `System` methods for stage flow
- `training.yaml`, `inference.yaml`, and `metrics.yaml`
- recipe-local `dataset/` or `src/` code

## Rough mental mapping

| SpeechBrain | ESPnet3 |
| --- | --- |
| `Brain` class | `System` + LightningModule + trainer wrapper |
| hyperparameter YAML | stage-specific YAML files |
| data prep script | `builder.py` |
| dataset pipeline | `dataset.py` plus `dataset:` config |
| train loop overrides | `src/lightning_module.py` or `src/trainer.py` |
| evaluation routine | `measure` stage |

## What usually carries over well

These ideas transfer well:

- custom PyTorch modules
- explicit data preparation
- recipe-local code
- careful experiment structure

## What changes

The main changes are:

- one monolithic training object is split into smaller layers
- inference is configured separately from training
- metrics are also their own stage
- publication is part of the recipe lifecycle

## Migration advice

Port one layer at a time:

1. port the dataset and collate behavior
2. port the model
3. port the optimizer and scheduler
4. port only the loop customizations you still need

If the old Brain class mixed several concerns together, separate them before
you port.

## Good pages to read next

<DocCards :cols="2">
  <DocCard
    title="Data and dataloader"
    desc="See the PyTorch-shaped explanation of datasets, collate functions, and builders."
    icon="tabler:database"
    href="../coming-from-pytorch/data-and-dataloader.html"
  />
  <DocCard
    title="Model and system"
    desc="See how model code and workflow code are separated in ESPnet3."
    icon="tabler:hierarchy-2"
    href="../coming-from-pytorch/model-and-system.html"
  />
  <DocCard
    title="System and stages"
    desc="See the architecture behind stage execution."
    icon="tabler:route"
    href="../../stages/index.html"
  />
  <DocCard
    title="Custom dataset"
    desc="See how to use ordinary PyTorch datasets in a recipe."
    icon="tabler:database"
    href="../finetuning/custom-dataset.html"
  />
  <DocCard
    title="Customize the training loop"
    desc="See when to replace the default LightningModule or trainer wrapper."
    icon="tabler:settings-cog"
    href="../finetuning/training-loop.html"
  />
</DocCards>
