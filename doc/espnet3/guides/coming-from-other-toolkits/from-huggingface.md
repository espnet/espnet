# From Hugging Face

If you come from Hugging Face, the main difference is scope.

- Hugging Face often starts from a model and a Trainer
- ESPnet3 often starts from a recipe, a System, and stage configs

Do not try to map every class one-to-one first.
Map the workflow first.

## Rough mental mapping

| Hugging Face | ESPnet3 |
| --- | --- |
| model class | `model` in `training.yaml`, often `src/model.py` |
| `Trainer` | `train` stage + Lightning trainer wrapper |
| dataset object | `dataset.py` or `builder.py` + `dataset:` config |
| preprocessing function | dataset transform, builder step, or collate function |
| generation config | `inference.yaml` |
| evaluation script | `metrics.yaml` + `measure` stage |
| model repo artifact | publication bundle |

## What usually feels familiar

These parts are usually easy to understand:

- normal PyTorch model code still works
- normal `torch.utils.data.Dataset` still works
- optimizer and scheduler config is explicit
- distributed training uses familiar backend settings under `trainer:`

## What usually feels different

These are the main shifts:

- config is split by stage, not by one big training script
- inference is a first-class stage, not just `generate()` calls
- recipes are expected to own data preparation too
- publication and demo flows are part of the system design

## A practical migration strategy

Move in this order:

1. get your dataset loading working
2. get your model instantiated from `training.yaml`
3. run one `train` stage locally
4. add `inference.yaml`
5. add `metrics.yaml` only after inference output looks right

Do not start by porting every helper utility.

## When to keep code recipe-local

Keep code under `src/` when it is specific to one recipe:

- `src/model.py`
- `src/system.py`
- `src/dataset.py`
- `src/trainer.py`
- `src/lightning_module.py`

Move code into `espnet3/` only when it is reusable across recipes.

## Good pages to read next

<DocCards :cols="2">
  <DocCard
    title="What is a recipe?"
    desc="See how ESPnet3 organizes one experiment as one recipe directory."
    icon="tabler:map"
    href="../../get-started/what-is-a-recipe.html"
  />
  <DocCard
    title="Coming from PyTorch"
    desc="See the closest mental model if you already understand raw PyTorch training code."
    icon="tabler:brand-pytorch"
    href="../coming-from-pytorch/index.html"
  />
  <DocCard
    title="Config overview"
    desc="See how `training.yaml`, `inference.yaml`, and `metrics.yaml` split the workflow."
    icon="tabler:settings-2"
    href="../../config/index.html"
  />
  <DocCard
    title="Custom dataset"
    desc="See how to plug in your own dataset without adopting a special dataset base class."
    icon="tabler:database"
    href="../finetuning/custom-dataset.html"
  />
  <DocCard
    title="Customize the model"
    desc="See how to use `src/model.py` and switch away from the task bridge when needed."
    icon="tabler:puzzle"
    href="../finetuning/custom-model.html"
  />
</DocCards>
