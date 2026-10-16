# Config diff

ESPnet3 is much more config-driven than ESPnet2.

If you are porting a recipe, this usually means:

- move shell variables into YAML
- split one large config surface into several files
- keep the model part mostly intact
- rewrite dataset, dataloader, and inference sections around ESPnet3 concepts

## The biggest change

In ESPnet2, you usually combine:

- shell options in `asr.sh`
- model config in `conf/train_asr_*.yaml`
- decode config in `conf/decode_asr.yaml`

In ESPnet3, those concerns are split across:

- `training.yaml`
- `inference.yaml`
- `metrics.yaml`
- `publication.yaml`
- `demo.yaml`

So the first migration step is not "rename one file".
It is "split one ESPnet2 config surface into several ESPnet3 configs".

## File mapping

| ESPnet2 | ESPnet3 |
| --- | --- |
| `conf/train_asr_*.yaml` | `conf/training.yaml` |
| `conf/decode_asr.yaml` | `conf/inference.yaml` |
| shell-side scoring options | `conf/metrics.yaml` |
| shell-side packing / upload settings | `conf/publication.yaml` |
| no direct equivalent | `conf/demo.yaml` |

## CLI flag mapping

| ESPnet2 shell side | ESPnet3 |
| --- | --- |
| `--asr_config` | `--training_config` |
| `--inference_config` | `--inference_config` |
| shell scoring stage options | `--metrics_config` |
| shell packing/upload options | `--publication_config` |
| no direct equivalent | `--demo_config` |

::: important
ESPnet3 expects canonical file and flag names.
Use `training.yaml`, `inference.yaml`, `metrics.yaml`, `publication.yaml`,
`demo.yaml`, and the matching `--*_config` flags.
:::

## What you can usually copy almost as-is

The safest part to port from ESPnet2 is the model definition itself.

For many ASR recipes, this means:

- `encoder`
- `encoder_conf`
- `decoder`
- `decoder_conf`
- `model_conf`
- `optim` / `optim_conf`
- `scheduler` / `scheduler_conf`

These often map directly into the `model`, `optimizer`, and `scheduler`
sections of `training.yaml`.

## Model migration

### ESPnet2 style

Typical ESPnet2 training config:

```yaml
encoder: transformer
encoder_conf:
  output_size: 256
  attention_heads: 4

decoder: transformer
decoder_conf:
  attention_heads: 4

model_conf:
  ctc_weight: 0.3

optim: adam
optim_conf:
  lr: 0.001

scheduler: warmuplr
scheduler_conf:
  warmup_steps: 2500
```

### ESPnet3 style

In ESPnet3, there are two main paths.

#### Reuse the ESPnet2 task path

This is the easiest migration path.

Set `task`, then put the old task-side model config under `model:`.

```yaml
task: espnet2.tasks.asr.ASRTask

model:
  encoder: transformer
  encoder_conf:
    output_size: 256
    attention_heads: 4

  decoder: transformer
  decoder_conf:
    attention_heads: 4

  model_conf:
    ctc_weight: 0.3
```

Then migrate optimizer and scheduler into Hydra-style sections:

```yaml
optimizer:
  _target_: torch.optim.Adam
  lr: 0.001

scheduler:
  _target_: espnet2.schedulers.warmup_lr.WarmupLR
  warmup_steps: 2500
```

This is usually the best first port.

#### Instantiate a pure ESPnet3 model directly

If you are no longer using the ESPnet2 task bridge, `model` becomes a direct
Hydra instantiation block:

```yaml
model:
  _target_: my_package.models.MyModel
  hidden_size: 256
```

But for ESPnet2-to-ESPnet3 migration, start with the first path unless you
already need a custom model wrapper.

<DocCards :cols="3">
  <DocCard
    title="Training Config"
    desc="See where model, optimizer, scheduler, dataloader, and trainer settings live."
    icon="tabler:settings-2"
    href="../../config/train_config.html"
  />
  <DocCard
    title="Model Components"
    desc="Compare the ESPnet2 task path with direct model instantiation."
    icon="tabler:cpu"
    href="../../core/components/model.html"
  />
  <DocCard
    title="Optimizer Config"
    desc="See Hydra-style optimizer and scheduler configuration."
    icon="tabler:adjustments"
    href="../../core/components/optimizer_configuration.html"
  />
</DocCards>

## Dataloader migration

This is the second big change.

In ESPnet2, you often have top-level fields such as:

- `batch_type`
- `batch_size`
- `accum_grad`
- sometimes shell-side `num_splits_*`

In ESPnet3, batching moves into `dataloader`.

### ESPnet2 style

```yaml
batch_type: folded
batch_size: 64
accum_grad: 1
max_epoch: 200
```

### ESPnet3 style

```yaml
dataloader:
  train:
    iter_factory:
      shuffle: true
      batches:
        type: folded
        shape_files:
          - ${stats_dir}/train/feats_shape
        batch_size: 64

trainer:
  accumulate_grad_batches: 1
  max_epochs: 200
```

### Practical rewrite rule

Use this mapping:

| ESPnet2 | ESPnet3 |
| --- | --- |
| `batch_type` | `dataloader.*.iter_factory.batches.type` |
| `batch_size` | `dataloader.*.iter_factory.batches.batch_size` |
| `accum_grad` | `trainer.accumulate_grad_batches` |
| `max_epoch` | `trainer.max_epochs` |

<DocCards :cols="3">
  <DocCard
    title="Dataloader"
    desc="See how iter_factory, samplers, batch samplers, and collate functions work."
    icon="tabler:layout-list"
    href="../../core/components/dataloader.html"
  />
  <DocCard
    title="Stats Collection"
    desc="See where feats_shape files come from before folded batching uses them."
    icon="tabler:gauge"
    href="../../stages/collect-stats.html"
  />
  <DocCard
    title="Train Stage"
    desc="See how training reads training.yaml and launches the trainer."
    icon="tabler:player-play"
    href="../../stages/train.html"
  />
</DocCards>

## Path and experiment settings

ESPnet3 makes many path settings explicit.

This is one of the biggest "extra config" additions compared with ESPnet2.

Typical `training.yaml` path scaffold:

```yaml
recipe_dir: .
data_dir: ${recipe_dir}/data
exp_tag: ${self_name:}
exp_dir: ${recipe_dir}/exp/${exp_tag}
stats_dir: ${recipe_dir}/exp/stats
dataset_dir: /path/to/your/dataset
```

ESPnet2 often kept these in:

- shell vars like `expdir`, `dumpdir`
- recipe-local assumptions
- stage-local defaults

ESPnet3 prefers making them visible in config.

### Main new path concepts

| ESPnet3 key | Why it exists |
| --- | --- |
| `recipe_dir` | anchor for local modules and relative paths |
| `exp_tag` | experiment naming |
| `exp_dir` | run output root |
| `stats_dir` | collect-stats outputs |
| `dataset_dir` | single shared dataset root across stages |
| `inference_dir` | inference output root |

<DocCards :cols="3">
  <DocCard
    title="Config Overview"
    desc="See how config files, defaults, and path resolvers fit together."
    icon="tabler:settings-code"
    href="../../config/index.html"
  />
  <DocCard
    title="System and Stages"
    desc="See which config object each stage receives."
    icon="tabler:hierarchy-2"
    href="../../stages/index.html"
  />
  <DocCard
    title="Recipe Structure"
    desc="See where conf, data, exp, and src files live in ESPnet3 recipes."
    icon="tabler:folder"
    href="./recipe-structure.html"
  />
</DocCards>

## Dataset migration

This is a major structural change.

In ESPnet2, data flow often depends on:

- `data/<split>/`
- `wav.scp`, `text`, `utt2spk`
- shell stage logic in `local/data.sh`

In ESPnet3, dataset access is described in config through `DataOrganizer`.

### ESPnet3 training dataset shape

```yaml
dataset:
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
  preprocessor:
```

### New concepts to learn

| ESPnet3 concept | Meaning |
| --- | --- |
| `data_src` | where the dataset class comes from |
| `data_src_args` | kwargs passed to `Dataset(...)` |
| `name` | logical test-set name |
| `preprocessor` | shared preprocessing layer |
| `recipe_dir` | allows local `dataset/__init__.py` resolution |

### Migration rule

If ESPnet2 used `local/data.sh` plus manifest-style files, the ESPnet3 version
usually becomes:

1. `dataset/builder.py` for source prep and manifest generation
2. `dataset/__init__.py` or `dataset/dataset.py` for `Dataset`
3. `dataset:` entries in `training.yaml` and `inference.yaml`

<DocCards :cols="3">
  <DocCard
    title="Dataset Config"
    desc="See the YAML format for train, valid, test, data_src, and data_src_args."
    icon="tabler:settings-2"
    href="../../core/components/data-organizer.html"
  />
  <DocCard
    title="DataOrganizer"
    desc="See how dataset entries become train, valid, and named test splits."
    icon="tabler:stack-2"
    href="../../core/components/data-organizer.html"
  />
  <DocCard
    title="Data Pipeline"
    desc="See how local/data.sh work maps to builder.py and dataset.py."
    icon="tabler:database"
    href="./data-pipeline.html"
  />
</DocCards>

## Trainer settings

ESPnet2 top-level training knobs often move into `trainer:`.

Typical examples:

| ESPnet2 | ESPnet3 |
| --- | --- |
| `max_epoch` | `trainer.max_epochs` |
| `accum_grad` | `trainer.accumulate_grad_batches` |
| distributed shell vars | `trainer.devices`, `trainer.num_nodes`, `trainer.strategy` |

So in ESPnet3, model-training parallelism belongs mainly to `trainer`, not to
the Dask `parallel` block.

<DocCards :cols="3">
  <DocCard
    title="Trainer"
    desc="See the ESPnet3 Lightning trainer wrapper and trainer config surface."
    icon="tabler:bolt"
    href="../../core/components/trainer.html"
  />
  <DocCard
    title="Multi-GPU Guide"
    desc="Compare trainer parallelism with provider/runner parallelism."
    icon="tabler:devices"
    href="../coming-from-pytorch/multi-gpu.html"
  />
  <DocCard
    title="Training Config"
    desc="See the trainer block inside the full training.yaml schema."
    icon="tabler:settings-2"
    href="../../config/train_config.html"
  />
</DocCards>

## Parallel config is new

ESPnet2 cluster behavior is usually shell-driven.
ESPnet3 adds an explicit `parallel:` config block.

Typical example:

```yaml
parallel:
  env: local
  n_workers: 1
```

Or:

```yaml
parallel:
  env: slurm
  n_workers: 16
  options:
    queue: batch
    cores: 4
    memory: 32GB
```

Use this for Dask-backed helper execution such as:

- `collect_stats`
- provider/runner workloads
- inference runner parallelism

Do not confuse it with Lightning DDP settings under `trainer`.

<DocCards :cols="3">
  <DocCard
    title="Parallel Config"
    desc="See env, n_workers, and backend options for local, GPU, and cluster execution."
    icon="tabler:binary-tree-2"
    href="../../core/parallel/provider_runner.html"
  />
  <DocCard
    title="Parallel Runtime"
    desc="See provider/runner execution for collect_stats, inference, and fan-out work."
    icon="tabler:arrows-split-2"
    href="../../core/parallel/index.html"
  />
  <DocCard
    title="Cluster Migration"
    desc="See what replaces nj, run.pl, queue.pl, and shell scheduler options."
    icon="tabler:server"
    href="./cluster-and-parallel.html"
  />
</DocCards>

## Inference is not "decode config" anymore

This is one of the most important differences.

In ESPnet2, `decode_asr.yaml` is often just decoding hyperparameters:

```yaml
beam_size: 10
ctc_weight: 0.3
lm_weight: 0.1
penalty: 0.0
```

In ESPnet3, `inference.yaml` is much broader.
It is not only beam-search settings.

It also defines:

- dataset test splits
- output directory
- provider
- runner
- input selection
- output formatting
- optional artifact writing
- optional parallel backend

### ESPnet3 inference shape

```yaml
recipe_dir: .
exp_tag:
exp_dir: ${recipe_dir}/exp/${exp_tag}
inference_dir: ${exp_dir}/${self_name:}

dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  recipe_dir: ${recipe_dir}
  test:
    - name: test
      data_src_args:
        split: test

parallel:
  env: local
  n_workers: 1

model:

input_key: speech
output_fn: src.inference.build_output

provider:
  _target_: espnet3.systems.base.inference_provider.InferenceProvider

runner:
  _target_: espnet3.systems.base.inference_runner.InferenceRunner
```

### The main migration rule for decoding

Think of ESPnet3 inference as:

- old decode config
- plus dataset selection
- plus output writing contract
- plus runtime execution backend

all combined in one file.

## What to do with old decode hyperparameters

Old ESPnet2 decode hyperparameters such as:

- `beam_size`
- `ctc_weight`
- `lm_weight`
- `penalty`
- `maxlenratio`
- `minlenratio`

usually do not disappear.

They normally move under the inference-side model config, because the model in
ESPnet3 is often instantiated directly from `inference.yaml`.

Conceptually:

```yaml
model:
  _target_: ...
  beam_size: 10
  ctc_weight: 0.3
  lm_weight: 0.1
  penalty: 0.0
```

So the beam-search knobs still exist, but they no longer define the whole
inference config by themselves.

<DocCards :cols="3">
  <DocCard
    title="Inference Config"
    desc="See how dataset, provider, runner, model, and outputs fit in inference.yaml."
    icon="tabler:wave-sine"
    href="../../config/infer_config.html"
  />
  <DocCard
    title="Inference Stage"
    desc="See how the infer stage reads inference.yaml and writes outputs."
    icon="tabler:player-play"
    href="../../stages/inference.html"
  />
  <DocCard
    title="Inference Provider"
    desc="See the provider contract used by parallel inference jobs."
    icon="tabler:route"
    href="../../core/parallel/inference_provider.html"
  />
</DocCards>

## decode vs inference

ESPnet2 naming often uses `decode`.
ESPnet3 docs and configs prefer `inference`.

Use these newer names when porting:

- `inference.yaml`, not `decode_asr.yaml`
- `inference_dir`, not `decode_dir`
- `infer` stage implementation, not ad-hoc decode shell logic

## Metrics config is also split out

In ESPnet2, scoring is often a continuation of decoding shell stages.

In ESPnet3, metric computation moves into `metrics.yaml`.

Typical shape:

```yaml
metrics:
  - metric:
      _target_: espnet3.systems.asr.metrics.wer.WER
      ref_key: ref
      hyp_key: hyp
```

This means:

- inference writes structured outputs first
- metrics read those outputs later

So scoring is no longer just a shell postprocess step.

<DocCards :cols="3">
  <DocCard
    title="Metrics Config"
    desc="See the metrics.yaml shape for metric classes and keys."
    icon="tabler:gauge"
    href="../../stages/metrics.html"
  />
  <DocCard
    title="Metrics Stage"
    desc="See how ESPnet3 runs metric computation after inference."
    icon="tabler:chart-bar"
    href="../../stages/metrics.html"
  />
  <DocCard
    title="Metrics Components"
    desc="See how metric classes are implemented and configured."
    icon="tabler:math-function"
    href="../../core/components/metrics.html"
  />
</DocCards>

## Publication and demo configs are extra

ESPnet3 also adds config surfaces that many ESPnet2 recipes did not expose as
first-class YAML files:

- `publication.yaml`
- `demo.yaml`

These exist because ESPnet3 treats:

- model packing
- model upload
- demo packing
- demo upload

as normal named stages with explicit config.

<DocCards :cols="3">
  <DocCard
    title="Publication Config"
    desc="See how model packing and upload settings are configured."
    icon="tabler:package"
    href="../../config/publish_config.html"
  />
  <DocCard
    title="Publish Stage"
    desc="See how ESPnet3 publishes packed models from stage config."
    icon="tabler:cloud-upload"
    href="../../stages/publish.html"
  />
  <DocCard
    title="Demo Config"
    desc="See the config surface for demo packaging and upload."
    icon="tabler:app-window"
    href="../../config/demo_config.html"
  />
</DocCards>

## A compact checklist

- Copy the old model/task-side config into `training.yaml:model`
- Convert `optim*` and `scheduler*` into Hydra-style blocks
- Move `batch_*` settings into `dataloader`
- Move epoch/accumulation/device settings into `trainer`
- Add explicit path scaffold keys
- Replace shell data assumptions with `dataset:`
- Rewrite decode config into a full `inference.yaml`
- Split scoring into `metrics.yaml`
- Add `parallel:` only when runner/Dask execution is needed

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Training Config"
    desc="See the full `training.yaml` structure used by create_dataset, stats, and train."
    icon="tabler:settings-2"
    href="../../config/train_config.html"
  />
  <DocCard
    title="Inference Config"
    desc="See how ESPnet3 inference combines dataset, provider, runner, and outputs."
    icon="tabler:wave-sine"
    href="../../config/infer_config.html"
  />
  <DocCard
    title="Metrics Config"
    desc="See how scoring moves into `metrics.yaml`."
    icon="tabler:gauge"
    href="../../stages/metrics.html"
  />
  <DocCard
    title="Dataset Config"
    desc="See the dataset reference format used by training and inference."
    icon="tabler:database"
    href="../../core/components/data-organizer.html"
  />
  <DocCard
    title="Parallel Config"
    desc="See the Dask-backed parallel config surface."
    icon="tabler:binary-tree-2"
    href="../../core/parallel/provider_runner.html"
  />
  <DocCard
    title="Recipe structure"
    desc="See how the file layout changes when porting a recipe."
    icon="tabler:folder"
    href="./recipe-structure.html"
  />
</DocCards>
