# Model and system

If you already know PyTorch, the good news is:

- your model can still just be a PyTorch module
- ESPnet3 does not force you to hide everything behind one fixed system class

The two main things ESPnet3 adds are:

- config-driven model instantiation
- a `System` class that owns stage flow

## The simple mental model

Think of the split like this:

- `src/model.py` -> your model code
- `training.yaml` -> how the model is instantiated
- `System` -> which stages run, and in what order

So if you come from PyTorch, ESPnet3 is not replacing your model.
It is wrapping the workflow around it.

## Custom model: src/model.py

The most direct pattern is to create:

```text
egs3/<recipe>/<system>/src/model.py
```

and put your normal PyTorch model there.

Minimal example:

```python
import torch


class MyModel(torch.nn.Module):
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.encoder = torch.nn.Linear(80, hidden_size)
        self.head = torch.nn.Linear(hidden_size, vocab_size)

    def forward(self, speech, text, **batch):
        hidden = self.encoder(speech)
        logits = self.head(hidden)
        loss = torch.nn.functional.cross_entropy(logits.transpose(1, 2), text)
        return loss, {"loss": loss.detach()}, speech.size(0)
```

That is just ordinary PyTorch.

## How to point config at the model

Use `model._target_` in `training.yaml`.

Example:

```yaml
task:

model:
  _target_: src.model.MyModel
  hidden_size: 256
  vocab_size: 500
```

This is the key point:

- leave `task` unset if you want direct model instantiation
- put the import path under `model._target_`

That is the clean ESPnet3 path for a custom model.

<DocCards :cols="3">
  <DocCard
    title="Model Components"
    desc="See the task bridge path and direct model._target_ path."
    icon="tabler:cpu"
    href="https://github.com/espnet/espnet/blob/master/espnet3/systems/base/training.py"
  />
  <DocCard
    title="Training Config"
    desc="See where task, model, optimizer, and trainer settings live."
    icon="tabler:settings-2"
    href="../../config/train_config.html"
  />
  <DocCard
    title="Custom Model"
    desc="See the finetuning-oriented recipe-local model guide."
    icon="tabler:puzzle"
    href="../finetuning/custom-model.html"
  />
</DocCards>

## What this replaces

If you stay on the old task bridge, `model:` is interpreted by the task-side
builder.

If you switch to `model._target_`, the recipe directly owns model
instantiation.

So the practical choice is:

- task bridge for old task-style models
- `model._target_` for recipe-local or custom models


## Why System exists

If you come from PyTorch, it is tempting to think only about the model.

But recipes often need workflow logic too:

- dataset preparation
- training
- inference
- metrics
- publication
- special recipe-local stages

That is what `System` owns.

So:

- model = computation
- system = workflow

<DocCards :cols="3">
  <DocCard
    title="System and Stages"
    desc="See how Systems own stage methods and config wiring."
    icon="tabler:hierarchy-2"
    href="../../stages/index.html"
  />
  <DocCard
    title="Stages"
    desc="See the built-in stage entrypoints and their config inputs."
    icon="tabler:route"
    href="../../stages/index.html"
  />
  <DocCard
    title="Recipe Structure"
    desc="See where src, conf, data, and exp files live in a recipe."
    icon="tabler:folder"
    href="../migrating-from-espnet2/recipe-structure.html"
  />
</DocCards>

## When a custom system is useful

A custom system becomes useful when the recipe needs behavior that is not just
"one standard train stage".

Examples:

- special preprocessing stage
- export stage
- multi-phase training
- curriculum training
- any workflow that needs multiple training passes with different configs

## Example: curriculum training with two training stages

Suppose you want:

1. easy training first
2. full training second

Then a clean ESPnet3 pattern is:

- create two training configs
- expose both configs in `run.py`
- add two system-specific stages

## Step 1: create two configs

For example:

```text
conf/
  training_easy.yaml
  training_full.yaml
```

The easy config might use:

- smaller subset
- shorter utterances
- easier curriculum
- fewer epochs

The full config then uses the full training setup.

## Step 2: declare the custom stages on the class

`StageSpec` declares a method name, config role and log directory. A subclass
inherits the tuple unchanged unless it replaces `stages`; declaring a tuple
containing only additions does not merge it with the parent's tuple.

```python
from espnet3.contract.stages import StageSpec
from espnet3.systems.esp2_asr.system import ASRSystem
from espnet3.systems.base.training import train


class CurriculumSystem(ASRSystem):
    # [:4] is create_dataset, train_tokenizer, collect_stats, train.
    # [4:] starts at infer and retains measurement/publication/demo stages.
    stages = (
        ASRSystem.stages[:4]
        + (
            StageSpec(name="train_easy", config="training", log_dir="exp_dir"),
            StageSpec(name="train_full", config="training", log_dir="exp_dir"),
        )
        + ASRSystem.stages[4:]
    )

    def train_easy(self):
        train(self.stage_configs["train_easy"])

    def train_full(self):
        train(self.stage_configs["train_full"])
```

These two methods use the same `training` role. They do not automatically
select different YAML files or initialize one phase from another. Put the
appropriate dataset, output directory and `fit.ckpt_path` in each phase's YAML.
Prepare the dataset/tokenizer/stats before calling these training helpers.
Every public method defined on a system must be declared in its `stages`.
Use `_private_names` for helpers that are not stages.

## Step 3: use the shared launcher

Save the class as `src/system.py` and make `run.py`:

```python
from src.system import CurriculumSystem
from espnet3.systems.launch import launch

if __name__ == "__main__":
    launch(CurriculumSystem, conf_package="egs3.TEMPLATE.esp2_asr")
```

The explicit package is important for a recipe-local subclass: the default
package is derived from the system's Python module, not its parent class.
`launch` derives stage choices and role flags from `StageSpec`; do not maintain
`ALL_STAGES`/`DEFAULT_STAGES` or add duplicate `--training_config` flags.

```bash
python run.py --stages train_easy --training_config conf/training_easy.yaml
python run.py --stages train_full --training_config conf/training_full.yaml
```

Both config files inherit the same TEMPLATE defaults. Keep their experiment
identities and checkpoint handoff explicit. This example calls the shared
training helper rather than `ASRSystem.train`, whose tokenizer prerequisite
requires its own stage config to be present in this baseline's CLI path.

<DocCards :cols="3">
  <DocCard
    title="Adding a Stage"
    desc="See how to expose recipe-local stages from a custom System."
    icon="tabler:route"
    href="../../contributing/adding-a-stage.html"
  />
  <DocCard
    title="Training Loop"
    desc="See when to use config, LightningModule, trainer, or System code."
    icon="tabler:player-play"
    href="../finetuning/training-loop.html"
  />
  <DocCard
    title="Config Overview"
    desc="See how stage-specific configs are loaded and passed to Systems."
    icon="tabler:settings-code"
    href="../../config/index.html"
  />
</DocCards>

## Why this is better than ad-hoc script logic

If you came from plain PyTorch, you might otherwise write:

- one Python script for easy training
- another Python script for full training

But in ESPnet3, it is often cleaner to keep one recipe and make the stage flow
explicit through the system.

That keeps:

- config
- logs
- stages
- output directories

under one consistent recipe structure.

## Good rule of thumb

Use:

- `src/model.py` when the model is custom
- a custom `System` when the workflow is custom

Do not use a custom system just because the model is custom.
Use it when the stage flow itself needs to change.

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Custom model"
    desc="See the finetuning-oriented version of recipe-local model customization."
    icon="tabler:puzzle"
    href="../finetuning/custom-model.html"
  />
  <DocCard
    title="Adding a stage"
    desc="See how to add recipe-local stages to a system."
    icon="tabler:route"
    href="../../contributing/adding-a-stage.html"
  />
  <DocCard
    title="Training loop"
    desc="See where to customize LightningModule, trainer, or stage flow."
    icon="tabler:player-play"
    href="../finetuning/training-loop.html"
  />
  <DocCard
    title="Training Config"
    desc="See how `model`, `optimizer`, `dataloader`, and `trainer` are configured."
    icon="tabler:settings-2"
    href="../../config/train_config.html"
  />
  <DocCard
    title="System and stages"
    desc="Read the architecture-level explanation of `System` and stage dispatch."
    icon="tabler:hierarchy-2"
    href="../../stages/index.html"
  />
</DocCards>
