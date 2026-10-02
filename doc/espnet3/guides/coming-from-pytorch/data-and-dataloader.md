# Data and dataloader

If you already know PyTorch, the dataset part of ESPnet3 is not very exotic.

The short version is:

::: important
Your recipe-local dataset can usually just be a normal
`torch.utils.data.Dataset`.
:::

ESPnet3 mainly adds:

- config-driven dataset resolution
- `DataOrganizer` for train/valid/test wiring
- builder logic for dataset preparation
- a dataloader config layer

## The part that stays familiar

If you know PyTorch, this still looks normal:

```python
from torch.utils.data import Dataset


class MyDataset(Dataset):
    def __init__(self, split: str):
        self.split = split
        self.samples = load_manifest(split)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            "speech": load_audio(sample["wav"]),
            "text": sample["text"],
        }
```

That is already a valid starting point for many ESPnet3 recipes.

## Where the dataset lives

Typical recipe-local layout:

```text
egs3/<recipe>/<system>/
  dataset/
    __init__.py
    dataset.py
    builder.py
```

`__init__.py` usually exports:

```python
from .dataset import MyDataset as Dataset
```

That is how `DataOrganizer` finds the local dataset by default.

## What builder.py is for

`builder.py` is the standard preparation hook used by `create_dataset`.

Use it for one-time preparation such as:

- download
- extraction
- manifest generation
- offline augmentation

So the simple rule is:

- sample loading -> `dataset.py`
- one-time preparation -> `builder.py`

<DocCards :cols="3">
  <DocCard
    title="Dataset References"
    desc="See the recipe-local Dataset and DatasetBuilder module contract."
    icon="tabler:folder-code"
    href="../../core/components/data-organizer.html"
  />
  <DocCard
    title="Create Dataset Stage"
    desc="See how dataset preparation is launched from a stage."
    icon="tabler:database-plus"
    href="../../stages/create-dataset.html"
  />
  <DocCard
    title="Data Pipeline"
    desc="See how recipe data prep maps to builder.py and dataset.py."
    icon="tabler:database"
    href="../migrating-from-espnet2/data-pipeline.html"
  />
</DocCards>

## What ESPnet3 adds on top of a plain dataset

The main extra layer is `DataOrganizer`.

It wires config into:

- `train`
- `valid`
- named `test` sets

So instead of manually constructing several datasets in Python code, ESPnet3
lets YAML define the split layout.

Minimal example:

```yaml
dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  recipe_dir: ${recipe_dir}
  train:
    - data_src_args:
        split: train
  valid:
    - data_src_args:
        split: valid
  test:
    - name: test
      data_src_args:
        split: test
```

If `data_src` is omitted, ESPnet3 loads the local recipe dataset module.

<DocCards :cols="3">
  <DocCard
    title="Dataset Config"
    desc="See the train, valid, test, data_src, and data_src_args format."
    icon="tabler:settings-2"
    href="../../core/components/data-organizer.html"
  />
  <DocCard
    title="DataOrganizer"
    desc="See how config entries become concrete dataset objects."
    icon="tabler:stack-2"
    href="../../core/components/data-organizer.html"
  />
  <DocCard
    title="Training Config"
    desc="See where dataset config sits in training.yaml."
    icon="tabler:player-play"
    href="../../config/train_config.html"
  />
</DocCards>

## Dataloader: slightly more ESPnet-specific

This is the part where ESPnet3 adds a bit more structure than plain PyTorch.

You still have normal concepts such as:

- batch size
- shuffle
- collate function

But they are usually expressed in config.

Minimal example:

```yaml
dataloader:
  collate_fn:
    _target_: espnet2.train.collate_fn.CommonCollateFn
    int_pad_value: -1

  train:
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: true
      collate_fn: ${dataloader.collate_fn}
      batches:
        type: unsorted
        batch_size: 4

  valid:
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: false
      collate_fn: ${dataloader.collate_fn}
      batches:
        type: unsorted
        batch_size: 4
```

<DocCards :cols="3">
  <DocCard
    title="Dataloader"
    desc="See iterator, batch sampler, collate, and logging details."
    icon="tabler:layout-list"
    href="../../core/components/dataloader.html"
  />
  <DocCard
    title="Stats Collection"
    desc="See how batch shapes and stats interact with dataloader behavior."
    icon="tabler:gauge"
    href="../../stages/collect-stats.html"
  />
  <DocCard
    title="Train Stage"
    desc="See how training consumes dataset and dataloader config."
    icon="tabler:player-play"
    href="../../stages/train.html"
  />
</DocCards>

## Why this is still nice for PyTorch users

Even though the loader is more config-driven, the split is still clean:

- dataset handles sample loading
- collate handles batch formatting
- config decides which dataset goes to which stage

So the mental model is still close to plain PyTorch.

## About CommonCollateFn

If your dataset returns ordinary sample dicts, `CommonCollateFn` is often still
a very useful default.

It can help with:

- automatic padding
- sequence length handling
- batch formatting that matches existing ESPnet model conventions

So a normal `Dataset` plus ESPnet collate is a good default combination.

## When you need the detailed docs

You usually do not need to learn the whole dataloader stack up front.

Start simple:

1. write a plain `Dataset`
2. make `training.yaml` point to it
3. keep `CommonCollateFn` if it already works
4. only then study the iterator/batching details

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Custom dataset"
    desc="Read the finetuning-oriented overview of recipe-local datasets and builders."
    icon="tabler:database"
    href="../finetuning/custom-dataset.html"
  />
  <DocCard
    title="Dataloader"
    desc="Read the detailed loader, collate, and iterator behavior."
    icon="tabler:layout-list"
    href="../../core/components/dataloader.html"
  />
  <DocCard
    title="DataOrganizer"
    desc="See how train, valid, and test datasets are assembled from YAML."
    icon="tabler:stack-2"
    href="../../core/components/data-organizer.html"
  />
  <DocCard
    title="Dataset references"
    desc="See how recipe-local dataset modules and builders are resolved."
    icon="tabler:folder-code"
    href="../../core/components/data-organizer.html"
  />
  <DocCard
    title="Dataset Config"
    desc="See the YAML dataset format used by training and inference."
    icon="tabler:settings-2"
    href="../../core/components/data-organizer.html"
  />
  <DocCard
    title="Training Config"
    desc="See where the dataloader and trainer config actually live."
    icon="tabler:player-play"
    href="../../config/train_config.html"
  />
</DocCards>
