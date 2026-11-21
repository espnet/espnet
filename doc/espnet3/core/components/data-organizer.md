---
title: 📦 ESPnet3 DataOrganizer
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 DataOrganizer

`DataOrganizer` is the config-driven wrapper that turns dataset entries into the
objects consumed by `collect_stats`, `train`, `infer`, and `measure`.

Implementation:

- `espnet3.components.data.data_organizer.DataOrganizer`
- Dataset reference resolution: `espnet3.components.data.dataset_module`
- Builder lifecycle: `espnet3.components.data.dataset_builder.DatasetBuilder`

## Role in the pipeline

```text
training.yaml / inference.yaml
  └── dataset:
        _target_: espnet3.components.data.data_organizer.DataOrganizer
        recipe_dir: ${recipe_dir}
        train: ...
        valid: ...
        test: ...
        preprocessor: ...
```

`DataOrganizer` then:

- builds `train` and `valid` as `CombinedDataset`
- builds each named `test` entry as `DatasetWithTransform`
- applies `transform`, then `preprocessor`

## Current dataset entry format

Each dataset entry uses:

- `data_src`
- `data_src_args`
- `transform`
- `name`

Minimal example:

```yaml
dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  recipe_dir: ${recipe_dir}

  train:
    - data_src: mini_an4/asr
      data_src_args:
        split: train

  valid:
    - data_src: mini_an4/asr
      data_src_args:
        split: valid

  test:
    - name: test
      data_src: mini_an4/asr
      data_src_args:
        split: test
```

Each entry must be a plain `dict`/`DictConfig` (the shape written above). The
`DatasetConfig` dataclass in `data_organizer.py` describes the same fields for
typing/reference purposes only — pass a real `DatasetConfig(...)` instance to
`train`/`valid`/`test` and construction fails, since the resolver expects a
mapping it can index by key, not a dataclass. Use YAML/dict entries, never a
`DatasetConfig` instance.

## Important behavior

**data_src_args only goes to Dataset(...)**

`DataOrganizer` resolves the dataset module, gets its exported `Dataset` class,
and instantiates:

```python
Dataset(**data_src_args)
```

Top-level organizer keys such as:

- `name`
- `transform`

stay in organizer space and are not forwarded to the dataset constructor.

**recipe_dir matters for local datasets**

If a dataset entry omits `data_src`, `DataOrganizer` resolves the local recipe
module:

```text
${recipe_dir}/dataset/__init__.py
```

So local recipes should set:

```yaml
dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  recipe_dir: ${recipe_dir}
```

**train and valid must move together**

Current `DataOrganizer` requires:

- both `train` and `valid` present, or
- both omitted

Providing only one side raises an error.

**test[*].name becomes the test-set key**

Inference and measurement use the `name` field to choose test sets and to build
per-test output directories.

**Dataset item contract: no utt_id key**

A recipe `Dataset.__getitem__` should return only the fields the
model/preprocessor actually consumes (e.g. `speech`, `text`). Do not add a
`utt_id` field to the sample dict — `CombinedDataset` passes the dict through
unchanged, so an extra key can reach a model/collate function that does not
expect it. The stable identifier for a sample is its `uid`: `CombinedDataset`
uses the sample's position (as a string) when every underlying dataset
supports integer indexing, or a dataset-provided string key otherwise (see
"String-indexed datasets" below). Inference/measurement code that needs an ID
should use that `uid`/index, not a field inside the sample.

**String-indexed datasets**

If any dataset in a split cannot be indexed by integer (probing `dataset[0]`
raises), `CombinedDataset` switches that whole split to string-index mode: it
collects string keys from every non-integer-indexable dataset (via `.keys()`
or by iterating the dataset) and looks samples up by key instead of position.
Duplicate keys across datasets in the same split raise `ValueError`.

## Transforms and preprocessor

Each dataset entry may define `transform`, and the organizer itself may define a
shared `preprocessor`.

The order is:

1. `transform(sample)`
2. `preprocessor(sample)` or `preprocessor(uid, sample)`

If the preprocessor is an `AbsPreprocessor`, ESPnet3 uses the ESPnet-style
`(uid, sample)` call automatically.

`DataOrganizer` also auto-injects `train=True`/`train=False` into an
`AbsPreprocessor`'s config so train/valid/test each get the right flag without
repeating it in YAML — but only when `preprocessor._target_` is a class that
subclasses `AbsPreprocessor` directly. If `_target_` points at a factory
*function* that returns an `AbsPreprocessor` instance, the flag is **not**
injected: whatever `train:` value is written in the config (or its default) is
used unchanged for every split. When using a factory function, have the
function itself accept and forward an explicit `train` argument per split
instead of relying on this auto-injection.

## Example: local dataset module

`mini_an4` uses the local recipe mode:

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
```

Because `data_src` is omitted, ESPnet3 loads:

```text
egs3/mini_an4/asr/dataset/__init__.py
```

and instantiates the exported `Dataset`.

## Example: tag-based dataset source

```yaml
test:
  - name: test-clean
    data_src: librispeech_100/asr
    data_src_args:
      split: test-clean
      recipe_dir: ${recipe_dir}
```

This resolves to:

```text
egs3.librispeech_100.asr.dataset
```

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Dataloader"
    desc="See how organized datasets feed the collate and iterator layer."
    icon="tabler:stack-2"
    href="./dataloader.html"
  />
  <DocCard
    title="Create dataset stage"
    desc="Return to the stage-level dataset creation flow."
    icon="tabler:route"
    href="../../stages/create-dataset.html"
  />
</DocCards>
