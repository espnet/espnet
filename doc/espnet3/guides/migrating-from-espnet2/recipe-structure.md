# Recipe structure

The directory layout changes a lot between ESPnet2 and ESPnet3.

This page shows:

1. a typical ESPnet2 recipe tree
2. a typical ESPnet3 recipe tree
3. how one maps to the other

## Typical ESPnet2 recipe tree

For a small ESPnet2 ASR recipe, the structure often looks like this:

```text
egs2/<recipe>/asr1/
  asr.sh
  run.sh
  cmd.sh
  path.sh
  conf/
    train_asr_*.yaml
    decode_asr.yaml
    slurm.conf
    queue.conf
  local/
    data.sh
    path.sh
    data_prep.py
  data/
  dump/
  exp/
```

This layout is strongly shell-oriented.

The main organizing files are usually:

- `asr.sh`
- `local/data.sh`
- `conf/train_asr_*.yaml`
- `conf/decode_asr.yaml`

## Typical ESPnet3 recipe tree

In ESPnet3, the canonical recipe layout is closer to this:

```text
egs3/<recipe>/<task>/
  run.py
  path.sh
  readme.md
  conf/
    training.yaml
    inference.yaml
    metrics.yaml
    publication.yaml
    demo.yaml
  dataset/
    __init__.py
    builder.py
    dataset.py
    config.yaml
  src/
    inference.py
    tokenizer.py
    preprocessor.py
    app.py
```

This layout is much more Python-centered (see `egs3/mini_an4/asr/` or
`egs3/librispeech_100/asr/` for the real tree).

The main organizing files are usually:

- `run.py`
- `conf/*.yaml`
- `dataset/builder.py`
- `dataset/dataset.py`
- `src/*.py`

## The biggest structural difference

ESPnet2 organizes the recipe around shell stages.

ESPnet3 organizes the recipe around:

- named Python stages
- stage-specific config files
- recipe-local Python modules

So the migration is not only a filename change.
It is a change in where logic is expected to live.

## Direct mapping

| ESPnet2 location | ESPnet3 location | Meaning |
| --- | --- | --- |
| `asr.sh` | `run.py` | top-level recipe entrypoint |
| shell stage numbers | named stages in `run.py` | workflow control |
| `conf/train_asr_*.yaml` | `conf/training.yaml` | training-side config |
| `conf/decode_asr.yaml` | `conf/inference.yaml` | inference-side config |
| shell scoring logic | `conf/metrics.yaml` + `measure` stage | metrics / scoring |
| shell packing / upload logic | `conf/publication.yaml` | model packaging and upload |
| no direct equivalent | `conf/demo.yaml` | demo packaging and upload |
| `local/data.sh` | `dataset/builder.py` | source prep and dataset build |
| recipe-local Python helpers under `local/` | `src/` | recipe-local Python utilities |
| `data/`, `dump/`, `exp/` conventions | explicit path keys in config | path scaffold is config-driven |

## asr.sh -> run.py

This is the most visible change.

### ESPnet2

`asr.sh` usually owns:

- numbered stages
- shell options
- cluster command wrappers
- decode loops
- stats / training / scoring orchestration

### ESPnet3

`run.py` owns:

- named stages
- config loading
- system construction
- stage dispatch

So the migration is roughly:

| ESPnet2 `asr.sh` concept | ESPnet3 `run.py` concept |
| --- | --- |
| `--stage`, `--stop_stage` | `--stages train infer measure` |
| shell stage blocks | system stage methods |
| shell config flags | `--training_config`, `--inference_config`, ... |

## local/data.sh -> dataset/builder.py

This is the second major move.

### ESPnet2

`local/data.sh` often owns:

- downloads
- extraction
- train/dev/test split
- manifest generation
- audio conversion

### ESPnet3

`dataset/builder.py` owns:

- `is_source_prepared()`
- `prepare_source()`
- `is_built()`
- `build()`

So the old shell data pipeline becomes a Python builder lifecycle.

## Recipe-local data loading

ESPnet2 often assumes prepared files under:

- `data/<split>/`
- `wav.scp`
- `text`
- `utt2spk`

ESPnet3 instead expects a recipe-local dataset module:

```text
dataset/
  __init__.py
  dataset.py
```

That module exports:

- `Dataset`
- optionally `DatasetBuilder`

So the mapping is:

| ESPnet2 | ESPnet3 |
| --- | --- |
| prepared Kaldi-style files | Python dataset object |
| shell conventions for splits | `dataset:` entries in YAML |
| task-local loader assumptions | `Dataset(...)` contract |

## local/ helpers -> src/

In ESPnet2, recipes often keep helper scripts in `local/`.

In ESPnet3, recipe-local Python helpers usually move into `src/`.

Typical examples:

- output formatting helpers
- custom inference wrappers
- publication README helpers
- Gradio demo app definitions

So:

| ESPnet2 | ESPnet3 |
| --- | --- |
| `local/*.py` helper | `src/*.py` helper |
| shell calls helper script | config or stage imports helper function |

## Config directory mapping

ESPnet2 often has one training config and one decode config.

ESPnet3 splits by pipeline area.

### ESPnet2

```text
conf/
  train_asr_transformer.yaml
  decode_asr.yaml
```

### ESPnet3

```text
conf/
  training.yaml
  inference.yaml
  metrics.yaml
  publication.yaml
  demo.yaml
```

The extra files exist because ESPnet3 turns more parts of the workflow into
normal config-driven stages.

## Path layout is less implicit

ESPnet2 often relies on shell variables such as:

- `dumpdir`
- `expdir`
- stage-local defaults

ESPnet3 prefers explicit config keys such as:

- `recipe_dir`
- `data_dir`
- `exp_tag`
- `exp_dir`
- `stats_dir`
- `dataset_dir`
- `inference_dir`

So one migration step is simply making old implicit paths explicit.

## What usually disappears

When moving to ESPnet3, these ESPnet2 files often disappear or shrink a lot:

- large shell stage bodies
- many `local/*.sh` wrappers
- decode-side shell loops
- stage-local SCP splitting glue

They are replaced by:

- config
- builder logic
- dataset logic
- provider/runner logic

## A good migration checklist

When porting a recipe tree, use this order:

1. `asr.sh` -> `run.py`
2. `train_asr_*.yaml` -> `training.yaml`
3. `decode_asr.yaml` -> `inference.yaml`
4. `local/data.sh` -> `dataset/builder.py`
5. dataset loading logic -> `dataset/dataset.py`
6. helper scripts -> `src/`
7. scoring / upload / demo -> `metrics.yaml`, `publication.yaml`, `demo.yaml`


## Related pages

<DocCards :cols="3">
  <DocCard
    title="Task to system"
    desc="See why ESPnet3 moved from task-centric ownership to systems and stages."
    icon="tabler:hierarchy-2"
    href="./task-to-system.html"
  />
  <DocCard
    title="Config diff"
    desc="See how ESPnet2 config surfaces split across ESPnet3 YAML files."
    icon="tabler:settings-2"
    href="./config-diff.html"
  />
  <DocCard
    title="Data pipeline"
    desc="See how `local/data.sh` maps to builders, datasets, and collate functions."
    icon="tabler:database"
    href="./data-pipeline.html"
  />
  <DocCard
    title="Cluster and parallel"
    desc="See how shell-driven recipe execution maps to provider and runner logic."
    icon="tabler:binary-tree-2"
    href="./cluster-and-parallel.html"
  />
</DocCards>
