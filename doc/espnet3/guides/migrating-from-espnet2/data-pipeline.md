# Data pipeline

The data pipeline changes a lot when moving from ESPnet2 to ESPnet3.

In ESPnet2, data preparation is usually centered on:

- `local/data.sh`
- one or more numbered shell stages in `asr.sh`
- Kaldi-style files under `data/<split>/`

In ESPnet3, the center of gravity moves into Python:

- `dataset/builder.py`
- `dataset/dataset.py`
- `dataset:` entries in YAML
- `DataOrganizer` and `DataLoaderBuilder`

## The main mental shift

In ESPnet2, data flow is often:

1. shell downloads or copies data
2. shell rewrites metadata
3. shell creates `wav.scp`, `text`, `utt2spk`, and friends
4. ESPnet reads those prepared files later

In ESPnet3, data flow is often:

1. `builder.py` checks and prepares source data
2. `builder.py` builds manifests or converted assets
3. `dataset.py` exposes a normal PyTorch dataset
4. YAML points `DataOrganizer` at that dataset

So the migration is not only "rewrite shell in Python".
It is also "make the dataset a real Python object".


## ESPnet3: builder.py owns preparation

In ESPnet3, the closest replacement is `dataset/builder.py`.

Typical layout:

```text
egs3/<recipe>/<task>/dataset/
  __init__.py
  builder.py
  dataset.py
  config.yaml
```

The builder lifecycle is:

- `is_source_prepared()`
- `prepare_source()`
- `is_built()`
- `build()`

This is the main replacement for old shell-side data stages.

## How to map old shell steps

| ESPnet2 | ESPnet3 |
| --- | --- |
| `local/data.sh` download stage | `prepare_source()` |
| shell extraction / copy step | `prepare_source()` |
| shell manifest generation | `build()` |
| shell train/dev split logic | `build()` |
| shell validation checks | `is_source_prepared()` / `is_built()` |

## Example: offline preparation in ESPnet3

`MiniAn4Builder` is a good small example.

Its `build()` method:

1. reads transcripts
2. converts SPH to WAV
3. splits train into train/valid
4. writes TSV manifests

The shape is roughly:

```python
from pathlib import Path

from espnet3.components.data.dataset_builder import DatasetBuilder
from espnet3.utils.download_utils import download_url, extract_targz


class MiniAn4Builder(DatasetBuilder):
    def is_source_prepared(self, recipe_dir: str | Path, **_kwargs) -> bool:
        source_dir = Path(recipe_dir) / "downloads"
        return (source_dir / "an4").is_dir()

    def prepare_source(self, recipe_dir: str | Path, **_kwargs) -> None:
        source_dir = Path(recipe_dir) / "downloads"
        source_dir.mkdir(parents=True, exist_ok=True)
        extract_targz(source_dir / "an4_raw.bigendian.tar.gz", source_dir)

    def is_built(self, recipe_dir: str | Path, **_kwargs) -> bool:
        data_dir = Path(recipe_dir) / "data"
        return all(
            (data_dir / split / "wav_text.tsv").is_file()
            for split in ("train", "valid", "test")
        )

    def build(self, recipe_dir: str | Path, **_kwargs) -> None:
        source_dir = Path(recipe_dir) / "downloads" / "an4"
        data_dir = Path(recipe_dir) / "data"

        rows = read_transcripts(source_dir)
        rows = convert_sph_to_wav(rows, data_dir / "wav")
        splits = split_train_valid_test(rows)

        for split, split_rows in splits.items():
            write_manifest(data_dir / split / "wav_text.tsv", split_rows)
```

Use the shared helpers in `espnet3.utils.download_utils` (`download_url`,
`extract_targz`) for downloads/extraction instead of ad-hoc HTTP or archive
code -- see the real `egs3/mini_an4/asr/dataset/builder.py` for the full
implementation.

`is_source_prepared()` and `is_built()` should be cheap filesystem checks.
`prepare_source()` and `build()` do the actual work.

That is exactly the kind of work that would often live in `local/data.sh` in
ESPnet2.

So if your old ESPnet2 recipe had a shell stage that:

- creates files once
- writes reusable manifests
- rewrites dataset layout on disk

the ESPnet3 home for that logic is usually `builder.py`.

## Offline augmentation

ESPnet2 often handled augmentation as an offline preparation step.

Typical examples are:

- speed perturbation stages
- audio rewriting stages
- text normalization written back to disk
- extra derived manifest files

In ESPnet3, the same kind of offline augmentation can still exist.

The natural place is `builder.py`, especially in `build()`.

Use this when the augmentation:

- should happen once
- should be cached on disk
- changes the reusable dataset assets themselves

Examples:

- resampling all audio files
- writing augmented manifests
- creating a derived dataset directory
- generating pseudo labels once

## Online augmentation is easier in ESPnet3

ESPnet3 also makes online augmentation much easier.

You can implement it directly in `dataset.py`.

That means augmentation can happen:

- when `__getitem__()` is called
- without rewriting files on disk
- with ordinary PyTorch logic

This is a big practical difference from ESPnet2.

### Typical online augmentation locations

In ESPnet3, online augmentation can live in:

- the dataset itself
- a per-entry `transform`
- a shared `preprocessor`

That gives you three useful layers:

1. dataset-local augmentation in `dataset.py`
2. split-specific transform in YAML
3. shared preprocessor in `DataOrganizer`

## Example: dataset-local online augmentation

Conceptually:

```python
class MyDataset(torch.utils.data.Dataset):
    def __getitem__(self, idx):
        sample = load_sample(idx)
        sample["speech"] = random_gain(sample["speech"])
        return sample
```

This is often the cleanest replacement for shell stages that existed only to
prepare one slightly modified variant of the same data.

## Dataset does not need to be an ESPnet-specific dataset anymore

This is another major change.

In ESPnet2, users often think in terms of `ESPnetDataset` and task-driven input
constraints.

In ESPnet3, recipe-local datasets can be ordinary `torch.utils.data.Dataset`
classes.

That means:

- custom indexing logic is easy
- custom in-memory caching is easy
- custom augmentation is easy
- custom sample dictionaries are easy

You are no longer forced into a special dataset abstraction just to plug into
the recipe.

## What dataset.py usually returns

ESPnet3 datasets usually return a Python dictionary per sample.

For example:

```python
{
    "speech": ...,
    "text": ...,
}
```

The exact keys depend on the rest of the pipeline.

## Return-value constraints: ESPnet2 model path vs custom model path

This is an important migration detail.

### If you still use an ESPnet2 task/model

If `training.yaml` sets `task` and reuses the ESPnet2 task bridge, then the
data still needs to satisfy the expectations of that task/model pair.

So in that case, the old ESPnet2-style constraints still matter:

- expected key names
- expected tensor shapes
- expected label fields

In other words, the dataset becomes more flexible, but the model side still
imposes structure.

### If you use a custom ESPnet3 model

If you instantiate your own model directly, the rule is looser.

Then the practical requirement is:

- the dataset must return whatever your model call path consumes

For training, that usually means the batch produced by the dataloader and
collate function must match what the model or Lightning wrapper expects.

For inference, it means `input_key` and `output_fn` must line up with the
dataset and model.

So the migration rule is:

- ESPnet2 task path -> keep task-side conventions
- custom model path -> define your own data contract clearly

## DataOrganizer replaces many old recipe-local conventions

`DataOrganizer` is the config-driven wrapper that turns dataset references into:

- `train`
- `valid`
- named `test` sets

Each entry can specify:

- `data_src`
- `data_src_args`
- `transform`
- `name`

That means the old shell convention of "this directory name implies this split"
is replaced by explicit config.

## Split-specific transforms

ESPnet3 can attach transforms directly in the dataset config.

Conceptually:

```yaml
dataset:
  train:
    - data_src_args:
        split: train
      transform:
        _target_: src.transforms.augment_training_sample
```

This is another place where online augmentation can live without shell stages.

## Collate function support is still strong

If you came from ESPnet2, you may worry that a plain PyTorch dataset means you
lose ESPnet conveniences.

You do not.

ESPnet3 can still use normal ESPnet collate functions such as:

- `espnet2.train.collate_fn.CommonCollateFn`

So even if your dataset is a normal `torch.utils.data.Dataset`, you can still get:

- automatic padding
- length calculation
- batch formatting that works well with ESPnet2-style models

This is one of the nicest parts of the migration story:

- flexible dataset implementation
- still compatible with useful ESPnet batching logic

## How to think about collate now

Use this split:

- dataset: sample-level logic
- collate_fn: batch-level padding and formatting

That is cleaner than pushing every data concern into shell-generated files.

## A practical migration rule

When reading an old ESPnet2 data pipeline, separate it into:

1. one-time source preparation
2. one-time artifact building
3. per-sample loading
4. per-sample augmentation
5. batch formatting

Then map them like this:

| Old concern | New home |
| --- | --- |
| download / extract | `prepare_source()` |
| manifest generation | `build()` |
| sample loading | `dataset.py` |
| online augmentation | `dataset.py` or `transform` |
| batch padding / lengths | `collate_fn` |

## When to keep augmentation offline

Prefer offline augmentation in `build()` when:

- the result should be reused across runs
- the transform is expensive
- the output should become a stable dataset artifact

Prefer online augmentation in `dataset.py` or `transform` when:

- randomness is useful every epoch
- you do not want extra disk copies
- the transform is naturally sample-local


## Related pages

<DocCards :cols="3">
  <DocCard
    title="Datasets"
    desc="Read the high-level dataset internals and recipe-local module layout."
    icon="tabler:database"
    href="../../core/components/data-organizer.html"
  />
  <DocCard
    title="Dataset references"
    desc="See how dataset modules, builders, and references are resolved."
    icon="tabler:folder-code"
    href="../../core/components/data-organizer.html"
  />
  <DocCard
    title="DataOrganizer"
    desc="See how train, valid, and named test sets are assembled from config."
    icon="tabler:stack-2"
    href="../../core/components/data-organizer.html"
  />
  <DocCard
    title="Dataloader"
    desc="See how collate, iter_factory, and batching are constructed."
    icon="tabler:layout-list"
    href="../../core/components/dataloader.html"
  />
  <DocCard
    title="Dataset Config"
    desc="See the YAML dataset entry format used by training and inference."
    icon="tabler:settings-2"
    href="../../core/components/data-organizer.html"
  />
  <DocCard
    title="Cluster and parallel"
    desc="See how heavy preparation loops can move from shell to provider/runner code."
    icon="tabler:binary-tree-2"
    href="./cluster-and-parallel.html"
  />
</DocCards>
