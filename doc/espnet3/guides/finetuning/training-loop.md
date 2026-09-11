# Customize the training loop

ESPnet3 training uses PyTorch Lightning.
For many recipes, `training.yaml` is enough.

You usually only need a custom training loop when:

- the step logic is unusual
- you need multiple optimizers
- you need GAN-style updates
- you need system-specific Trainer setup
- you need multiple training stages such as curriculum training

## Default path

The shared training path already gives you:

- `ESPnetLightningModule`
- `ESPnet3LightningTrainer`
- config-driven optimizer and scheduler setup
- config-driven callbacks, logger, and profiler

If that fits your recipe, keep using it.

::: warning
Custom `src/lightning_module.py` and `src/trainer.py` are not auto-discovered
by the current training path.

Today, if you want to use a custom LightningModule or Trainer, you also need to
wire that path explicitly from recipe-local `System` code or another training
entrypoint override.

ESPnet3 TODO: make this customization path easier and more direct in the
future.
:::

## Customize src/lightning_module.py

Add `src/lightning_module.py` when you need to change step logic itself.

Typical reasons:

- custom `training_step`
- custom `validation_step`
- manual optimization
- named `OptimizationStep` updates

Minimal sketch:

```python
from espnet3.components.modeling.lightning_module import ESPnetLightningModule


class MyLightningModule(ESPnetLightningModule):
    def training_step(self, batch, batch_idx):
        loss, stats, weight = self.model(**batch)
        self.log("train/loss", loss)
        return loss
```

This file alone does not change the active training path.
You still need recipe-local wiring so training uses `MyLightningModule`
instead of the shared `ESPnetLightningModule`.

## Customize src/trainer.py

Add `src/trainer.py` when you need to change Trainer construction.

Typical reasons:

- system-specific callback setup
- system-specific config normalization
- extra validation around Trainer options

Minimal sketch:

```python
from espnet3.components.trainers.trainer import ESPnet3LightningTrainer


class MyTrainer(ESPnet3LightningTrainer):
    def __init__(self, model=None, exp_dir=None, config=None, best_model_criterion=None):
        super().__init__(
            model=model,
            exp_dir=exp_dir,
            config=config,
            best_model_criterion=best_model_criterion,
        )
```

This file alone does not change the active training path either.
You still need recipe-local wiring so training uses `MyTrainer` instead of the
shared `ESPnet3LightningTrainer`.

## How to actually use them

Today, the practical way is to wire both from recipe-local `System` code.

Minimal sketch:

```python
from espnet3.systems.asr.system import ASRSystem
from espnet3.systems.base.training import _instantiate_model

from src.lightning_module import MyLightningModule
from src.trainer import MyTrainer


class MySystem(ASRSystem):
    def train(self):
        config = self.training_config
        model = _instantiate_model(config)
        lit_model = MyLightningModule(model, config)
        trainer = MyTrainer(
            model=lit_model,
            exp_dir=config.exp_dir,
            config=config.trainer,
            best_model_criterion=config.best_model_criterion,
        )
        trainer.fit()
```

If you also need custom collect-stats behavior, wire that path in the same
place.

## Recipe-local code is fine

Put recipe-specific code under `src/` first.

```text
egs3/<recipe>/<system>/
  src/
    model.py
    lightning_module.py
    trainer.py
    system.py
```

Only move the code into `espnet3/` when it is clearly reusable across recipes.
For example, a GAN helper that only one recipe uses can stay recipe-local.


## Related pages

<DocCards :cols="3">
  <DocCard
    title="Customize the model"
    desc="See when `src/model.py` is enough and when you also need loop changes."
    icon="tabler:puzzle"
    href="./custom-model.html"
  />
  <DocCard
    title="Adding a stage"
    desc="See how to add custom stages and extra configs through `run.py`."
    icon="tabler:route"
    href="../../contributing/adding-a-stage.html"
  />
  <DocCard
    title="Trainer component"
    desc="Read the shared Lightning trainer wrapper behavior."
    icon="tabler:settings-cog"
    href="../../core/components/trainer.html"
  />
  <DocCard
    title="Multiple optimizers"
    desc="Read the contract used for named optimizer updates."
    icon="tabler:git-merge"
    href="../../core/components/multiple_optimizers_schedulers.html"
  />
  <DocCard
    title="Training config"
    desc="See where trainer, optimizer, scheduler, and profiler are configured."
    icon="tabler:settings-2"
    href="../../config/train_config.html"
  />
</DocCards>
