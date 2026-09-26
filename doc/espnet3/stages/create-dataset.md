---
title: ESPnet3 Create Dataset Stage
author:
- name: "Masao Someki"
- name: "Elias Naske"
date: 2026-05-14
---

# ESPnet3 Create Dataset Stage

`create_dataset()` is responsible for downloading and preparing datasets.

For each unique dataset source defined for a partition (`dataset.train`, `dataset.valid`, and `dataset.test` in `training.yaml`), it does the following:

1. resolves the dataset module
2. instantiates a [builder](#builder) object.
3. prepares source files, if needed
4. build the dataset, if needed

The same dataset source is only prepared once per stage run.

## 1. Run

```bash
python run.py --stages create_dataset --training_config conf/training.yaml
```

### Where the code lives

Within a typical recipe structure, the necessary files are organized like so:

```text
egs3/<recipe>/<task>/
├── conf/
│   └── training.yaml   # config file
└── dataset/
    ├── __init__.py     # exports Dataset and Builder classes
    ├── builder.py      # handles source preparation and build-time side effects
    └── dataset.py      # defines the Dataset class used for training and inference.
```

## 2. Configuration

The `create_dataset` stage is configured from the `create_dataset` block in
`training.yaml`.

The following keys are available:

| Key           | Description       | Example                           |
| ------------- | ----------------- | --------------------------------- |
| `recipe_dir`  | Recipe directory  | `egs3/mini_an4/asr`               |
| `dataset_dir` | Dataset directory | `egs3/mini_an4/asr/data/mini_an4` |

For more information, see [Training Configuration](../config/train_config.html)

::: warning
`create_dataset` does **not** support a `func`-style callable hook. The `create_dataset` block in
[`egs3/TEMPLATE/asr/conf/training.yaml`](https://github.com/espnet/espnet/blob/master/egs3/TEMPLATE/asr/conf/training.yaml)
ships with a `func: src.creating_dataset.create_dataset` field by default, but
`BaseSystem.create_dataset()`
([`espnet3/systems/base/system.py`](https://github.com/espnet/espnet/blob/master/espnet3/systems/base/system.py))
never reads `func` — any key you put under `create_dataset:` is simply forwarded as a keyword
argument to the builder methods described [below](#3-builder). Delete `func` from a copied
`training.yaml` (or ignore it — a recipe builder written with `**_kwargs` silently absorbs it) and
implement dataset preparation through the `DatasetBuilder` contract instead.
:::

### Example

```yaml
dataset_dir: ${recipe_dir}/data/mini_an4

create_dataset:
  recipe_dir: ${recipe_dir}
  dataset_dir: ${dataset_dir}

dataset:
  train:
    - name: train
      data_src: mini_an4/asr
      data_src_args:
        split: train
        data_path: ${dataset_dir}
  valid:
    - name: valid
      data_src: mini_an4/asr
      data_src_args:
        split: valid
        data_path: ${dataset_dir}
```



## 3. Builder

The code for preparding the dataset is defined in a builder class in `builder.py`, which inherits from [`espnet3.components.data.DatasetBuilder`](https://github.com/espnet/espnet/blob/master/espnet3/components/data/dataset_builder.py).

The builder has two main responsibilities:
- **Source Preparation**: download, extract, validate, or locate raw assets
- **Building**: run task-ready preprocessing or other recipe-local dumping data

To implement these, the builder class must define the following methods:

| Method                         | Returns | What it does                                                                                            |
| ------------------------------ | ------- | ------------------------------------------------------------------------------------------------------- |
| `is_source_prepared(**kwargs)` | `bool`  | Checks if the raw dataset source files are already available. If `True`, `prepare_source()` is skipped. |
| `prepare_source(**kwargs)`     | `None`  | Prepares raw source files.                                                                              |
| `is_built(**kwargs)`           | `bool`  | Checks if manifest files are already built. If `True`, `build()` is skipped.                            |
| `build(**kwargs)`              | `None`  | Builds manifest files for each partition, preprocesses audio files, etc.                                |

The arguments passed to the builder methods come from the `create_dataset`
block in `training.yaml`.
For example, if the config looks like this:
```yaml
create_dataset:
  recipe_dir: ${recipe_dir}
  source_dir: ${dataset_dir}
```

The builder will be called like this:

```python
if not builder.is_source_prepared(recipe_dir=..., source_dir=...):
  builder.prepare_source(recipe_dir=..., source_dir=...)

if not builder.is_built(recipe_dir=..., source_dir=...):
  builder.build(recipe_dir=..., source_dir=...)
```

## 4. `dataset/__init__.py`

To ensure that the dataset and builder classes are accessible to other modules, they should be exported in `dataset/__init__.py` as `Dataset` and `DatasetBuilder`, respectively.

Minimal example:

```python
from egs3.my_recipe.asr.dataset.builder import MyDatasetBuilder as DatasetBuilder
from egs3.my_recipe.asr.dataset.dataset import MyDataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
```

## 5. How dataset modules are resolved

Dataset resolution is shared with the normal dataset loading path:

1. `data_src: mini_an4/asr`
2. `data_src: egs3.mini_an4.asr.dataset`
3. omitted `data_src`, which loads `${recipe_dir}/dataset/__init__.py`

## 6. Examples

### Example 1: `mini_an4`

`egs3/mini_an4/asr/dataset/builder.py` is a full build example.

Behavior:

- `prepare_source()` extracts the AN4 archive under the recipe dataset area
- `build()` converts audio and writes manifest TSVs under `data/manifest/`

The resulting tree is roughly:

```text
egs3/mini_an4/asr/
└── data/
    ├── manifest/
    │   ├── train.tsv
    │   ├── valid.tsv
    │   └── test.tsv
    └── wav/
        ├── train/
        └── test/
```

Minimal conceptual export in `__init__.py`:

```python
from egs3.mini_an4.asr.dataset.builder import MiniAn4Builder as DatasetBuilder
from egs3.mini_an4.asr.dataset.dataset import MiniAn4Dataset as Dataset

__all__ = ["Dataset", "DatasetBuilder"]
```

### Example 2: `librispeech_100`

`egs3/librispeech_100/asr/dataset/builder.py` is the contrasting pattern.

Behavior:

- `prepare_source()` only validates that the LibriSpeech tree exists
- `is_built()` simply reuses source readiness
- `build()` is effectively a no-op validation path

This recipe reads the original corpus layout directly instead of generating
separate manifests.

This is the contrasting pattern to `mini_an4`: the builder still participates
in the stage lifecycle, but the recipe chooses not to materialize a separate
manifest representation.

## Notes

- `create_dataset` should be deterministic and safe to re-run
- source preparation and build are intentionally separate checks
- the same dataset source is only prepared once even if it appears in multiple
  splits

## Related pages

**Stage API:** [`DataOrganizer`](../../guide/espnet3/components/DataOrganizer.html),
[`DatasetBuilder`](../../guide/espnet3/components/DatasetBuilder.html), and
[`instantiate_dataset_reference`](../../guide/espnet3/components/instantiate_dataset_reference.html).
Continue with [collect_stats](./collect-stats.html) after dataset preparation.

<DocCards :cols="3">
  <DocCard
    title="Dataset references and builders"
    desc="See the dataset-side builders and references used before batching."
    icon="tabler:database"
    href="../core/components/data-organizer.html"
  />
  <DocCard
    title="DataOrganizer"
    desc="See how dataset split are organized before batching."
    icon="tabler:folders"
    href="../core/components/data-organizer.html"
  />
  <DocCard
    title="Training dataset config"
    desc="See how dataset options are configured in training.yaml"
    icon="tabler:puzzle"
    href="../core/components/data-organizer.html"
  />
</DocCards>
