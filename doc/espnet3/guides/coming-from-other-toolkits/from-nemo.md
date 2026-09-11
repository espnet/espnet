# From NeMo

If you come from NeMo, the biggest shift is that ESPnet3 is less
framework-centric and more recipe-centric.

NeMo often centers around:

- task collections and model classes
- config-driven training with a framework-owned runtime
- deployment/export paths around the model artifact

ESPnet3 centers around:

- one recipe directory
- one System that owns stages
- separate configs for training, inference, metrics, publication, and demo

## Rough mental mapping

| NeMo | ESPnet3 |
| --- | --- |
| model module | `src/model.py` or shared `espnet3` model |
| training config | `training.yaml` |
| inference or decode config | `inference.yaml` |
| data module | `dataset:` + `dataloader:` + `builder.py` |
| experiment manager behavior | `trainer:` + callbacks + logging config |
| export or deploy artifact | publication bundle and demo package |

## What to expect

ESPnet3 usually asks you to be more explicit about:

- stage boundaries
- file layout
- dataset preparation
- where outputs are written

That is a feature.
It makes recipes easier to inspect and publish later.

## Migration advice

Do not port the whole NeMo project structure first.
Instead:

1. port the dataset path
2. port the model
3. port optimizer and scheduler config
4. run one local train stage
5. add inference and metrics after training works

## When to customize deeply

If your old NeMo project has custom optimization logic, GAN training, or
non-standard loop behavior, do not force everything into one model class.

ESPnet3 lets you split that work across:

- `src/model.py`
- `src/lightning_module.py`
- `src/trainer.py`
- `src/system.py`

## Good pages to read next

<DocCards :cols="2">
  <DocCard
    title="System and stages"
    desc="See the main architectural unit behind ESPnet3 recipe execution."
    icon="tabler:route"
    href="../../stages/index.html"
  />
  <DocCard
    title="Config overview"
    desc="See the stage-specific config split used across ESPnet3."
    icon="tabler:settings-2"
    href="../../config/index.html"
  />
  <DocCard
    title="Custom dataset"
    desc="See how data loading is organized in ESPnet3."
    icon="tabler:database"
    href="../finetuning/custom-dataset.html"
  />
  <DocCard
    title="Customize the model"
    desc="See how to wire a custom model and custom training logic."
    icon="tabler:puzzle"
    href="../finetuning/custom-model.html"
  />
  <DocCard
    title="Customize the training loop"
    desc="See when to use recipe-local LightningModule or trainer wrappers."
    icon="tabler:settings-cog"
    href="../finetuning/training-loop.html"
  />
</DocCards>
