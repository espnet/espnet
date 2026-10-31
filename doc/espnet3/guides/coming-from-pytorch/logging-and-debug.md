# Logging and debug

If you come from plain PyTorch, ESPnet3 already logs more than a small custom
training script usually does.

The main idea is:

- stage logs are written automatically
- configs are dumped into logs
- trainer/logger/profiler are configured from YAML
- Lightning features such as W&B and `fast_dev_run` are available

## First place to look

When something is wrong, start with the stage log under the relevant output
directory.

Typical places are:

- `exp/.../train.log`
- `exp/.../inference/.../infer.log`
- `exp/.../inference/.../measure.log`

ESPnet3 switches the stage log handler automatically when stages run.

<DocCards :cols="3">
  <DocCard
    title="System and Stages"
    desc="See how stage execution selects log locations."
    icon="tabler:hierarchy-2"
    href="../../stages/index.html"
  />
  <DocCard
    title="Train Stage"
    desc="See where training logs and config inputs come from."
    icon="tabler:player-play"
    href="../../stages/train.html"
  />
  <DocCard
    title="Inference Stage"
    desc="See how inference writes per-test-set outputs and logs."
    icon="tabler:wave-sine"
    href="../../stages/inference.html"
  />
</DocCards>

## Logging utilities

ESPnet3 logging is built around a small utility layer in
`espnet3.utils.logging_utils`.

The usual entrypoints are:

| Utility | Role |
| --- | --- |
| `configure_logging()` | configure console logging and optional run log file |
| `set_stage_log_handler()` | switch the active stage log file |
| `log_stage()` | add the current stage label to log records |
| `log_stage_metadata()` | log command, config paths, environment, and resolved config |
| `log_run_metadata()` | log command, Python, git, and optional requirements metadata |
| `log_env_metadata()` | log CUDA, distributed, runtime, and scheduler environment |

For most recipes, you do not call these manually.
`run_stages()` calls them before each stage runs.

Minimal manual setup looks like this:

```python
import logging
from pathlib import Path

from espnet3.utils.logging_utils import configure_logging, log_run_metadata


logger = configure_logging(log_dir=Path("exp/my_run"), level=logging.INFO)
log_run_metadata(logger, write_requirements=True)
logger.info("custom setup done")
```

That creates console output and `exp/my_run/run.log`.

<DocCards :cols="3">
  <DocCard
    title="configure_logging"
    desc="See the root logger, console handler, run.log, and warning capture setup."
    icon="tabler:terminal"
    href="../../../guide/espnet3/utils/configure_logging.html"
  />
  <DocCard
    title="log_run_metadata"
    desc="See command, Python, git, config path, and requirements logging."
    icon="tabler:clipboard-list"
    href="../../../guide/espnet3/utils/log_run_metadata.html"
  />
  <DocCard
    title="log_env_metadata"
    desc="See CUDA, distributed, runtime, and scheduler environment logging."
    icon="tabler:server-cog"
    href="../../../guide/espnet3/utils/log_env_metadata.html"
  />
</DocCards>

## Automatic log rotation

ESPnet3 rotates logs when a target file already exists.
This is startup rotation, not size-based rotation.

For example, if `run.log` already exists:

```text
exp/my_run/
  run.log
```

then the next `configure_logging(log_dir=Path("exp/my_run"))` call moves the
old file and creates a fresh current log:

```text
exp/my_run/
  run.log      # current run
  run1.log     # previous run
```

If `run1.log` already exists, the old file becomes `run2.log`, and so on.
The same rule is used for stage logs such as `train.log` and `infer.log`.

Stage logging uses one active stage file handler at a time.
When the stage changes, `set_stage_log_handler()` removes the previous stage
handler, rotates the next target if needed, and attaches the new handler.

```text
exp/asr_train/
  train.log
  train1.log

exp/asr_train/inference/test/
  infer.log
  infer1.log
```

For distributed training, the `train` stage defaults to `rank0` file logging.
Non-zero ranks skip the stage file handler to avoid multiple processes rotating
the same file.

Set `stage_log_mode: per_rank` in the training config when you need one file
per rank:

```yaml
stage_log_mode: per_rank
```

That writes files such as:

```text
exp/asr_train/
  train_rank0.log
  train_rank1.log
```

<DocCards :cols="3">
  <DocCard
    title="set_stage_log_handler"
    desc="See how stage log handlers are replaced and rotated."
    icon="tabler:file-analytics"
    href="../../../guide/espnet3/utils/set_stage_log_handler.html"
  />
  <DocCard
    title="log_stage"
    desc="See how the stage label is attached to each log record."
    icon="tabler:tag"
    href="../../../guide/espnet3/utils/log_stage.html"
  />
  <DocCard
    title="log_stage_metadata"
    desc="See what metadata is written at the start of each stage."
    icon="tabler:notes"
    href="../../../guide/espnet3/utils/log_stage_metadata.html"
  />
</DocCards>

## What ESPnet3 logs automatically

### Run metadata

ESPnet3 logs:

- start timestamp
- full command line
- Python executable
- Python version
- working directory
- config file paths
- git commit / branch / dirty state

If requested, it also writes:

- `requirements.txt`

next to the log file.

## 2. Environment metadata

ESPnet3 also logs environment information such as:

- `CUDA_VISIBLE_DEVICES`
- `RANK`, `LOCAL_RANK`, `WORLD_SIZE`
- `MASTER_ADDR`, `MASTER_PORT`
- `PATH`, `PYTHONPATH`, `LD_LIBRARY_PATH`
- cluster scheduler variables such as `SLURM_*`, `PBS_*`, `LSF_*`
- runtime variables such as `NCCL_*`, `CUDA_*`, `OMP_*`, `MKL_*`

This is very useful when debugging multi-node or cluster issues.

## 3. Resolved config contents

Per-stage metadata logging also dumps resolved config YAML into the log.

That includes, when present:

- `training_config`
- `inference_config`
- `metrics_config`

So if you want to know what config actually ran, the log already contains it.

## 4. Component structure

ESPnet3 logs several instantiated components explicitly.

Examples include:

- dataloader details
- callback objects
- provider env objects
- metric objects
- model-side components in some paths

This is useful when checking whether the config instantiated the object you
expected.

## 5. Training metrics

The default training callbacks log:

- batch-level summaries every `log_every_n_steps`
- end-of-epoch train summaries
- end-of-epoch validation summaries
- learning rates

The summary lines include timings such as:

- `iter_time`
- `forward_time`
- `backward_time`
- `optim_step_time`
- `train_time`
- `valid_time`

So you already get some lightweight performance visibility without adding your
own callback.

## Default training callbacks

ESPnet3 adds these by default:

- last checkpoint saving
- best-k checkpoint saving
- averaged best-k checkpoint saving
- learning-rate monitor
- metrics logger
- TQDM progress bar

So even a small recipe already has a more opinionated logging setup than a
minimal raw PyTorch loop.

## Logger configuration

Training logger configuration lives under `trainer.logger`.

For example, TensorBoard:

```yaml
trainer:
  logger:
    - _target_: lightning.pytorch.loggers.TensorBoardLogger
      save_dir: ${exp_dir}/tensorboard
      name: tb_logger
```

Because ESPnet3 uses Lightning here, you can also use other Lightning loggers.

For example, W&B:

```yaml
trainer:
  logger:
    - _target_: lightning.pytorch.loggers.WandbLogger
      project: my-espnet3-project
      name: ${exp_tag}
      save_dir: ${exp_dir}
```

Other common options include:

- `CSVLogger`
- `TensorBoardLogger`
- `WandbLogger`

So if you already know Lightning logger config, it transfers directly.

<DocCards :cols="3">
  <DocCard
    title="Training Config"
    desc="See trainer.logger, profiler, and fast debug settings in YAML."
    icon="tabler:settings-2"
    href="../../config/train_config.html"
  />
  <DocCard
    title="Trainer"
    desc="See the trainer wrapper and its Lightning integration points."
    icon="tabler:bolt"
    href="../../core/components/trainer.html"
  />
  <DocCard
    title="Callbacks"
    desc="See default checkpoint, LR, progress, and metrics callbacks."
    icon="tabler:bell"
    href="../../core/components/callbacks.html"
  />
</DocCards>

## Profiler configuration

Profiler settings live under `trainer.profiler`.

Because this is passed to Lightning, you can use Lightning profiler configs
directly.

Example:

```yaml
trainer:
  profiler:
    _target_: lightning.pytorch.profilers.SimpleProfiler
    dirpath: ${exp_dir}
    filename: profiler.txt
```

Or:

```yaml
trainer:
  profiler:
    _target_: lightning.pytorch.profilers.AdvancedProfiler
    dirpath: ${exp_dir}
    filename: profiler.txt
```

Use this when:

- training is unexpectedly slow
- validation is much slower than expected
- you want a first pass before deeper PyTorch profiling

## Fast debug knobs

If you just want to see whether the whole training path runs once, Lightning
already gives you useful debug flags through `trainer`.

### fast_dev_run

This is the fastest sanity-check setting.

```yaml
trainer:
  fast_dev_run: true
```

That runs a tiny end-to-end debug pass so you can confirm:

- dataset works
- collate works
- model forward works
- validation step works
- logging setup works

This is one of the best first checks when wiring a new recipe.

### Other useful trainer-side debug settings

Examples:

```yaml
trainer:
  limit_train_batches: 2
  limit_val_batches: 2
  max_epochs: 1
```

These are often easier than editing the dataset itself for quick checks.

## Good debugging order

When something breaks, use this order:

1. run with `fast_dev_run: true`
2. inspect the stage log
3. confirm the resolved config dump
4. confirm the instantiated dataloader / callbacks / logger
5. only then add profiler settings if the issue is performance

## What to check for performance issues

For performance debugging, the built-in summary logs already tell you whether
time is going into:

- iteration
- forward
- backward
- optimizer step
- validation

If that is not enough, add a profiler under `trainer.profiler`.

For cluster or multi-GPU issues, also inspect the logged environment variables.

<DocCards :cols="3">
  <DocCard
    title="Multi-GPU"
    desc="See how trainer parallelism differs from provider/runner parallelism."
    icon="tabler:gpu-card"
    href="./multi-gpu.html"
  />
  <DocCard
    title="Parallel Config"
    desc="See cluster and worker settings that show up in logs."
    icon="tabler:binary-tree-2"
    href="../../core/parallel/provider_runner.html"
  />
  <DocCard
    title="Dataloader"
    desc="See sampler and iterator behavior when performance issues start in loading."
    icon="tabler:layout-list"
    href="../../core/components/dataloader.html"
  />
</DocCards>

## Why this feels different from plain PyTorch

In plain PyTorch, you often write:

- your own logger setup
- your own metric summaries
- your own profiler plumbing
- your own W&B integration

In ESPnet3, much of that is already structured through:

- stage logging
- Lightning trainer config
- default callbacks
- config-driven logger/profiler objects

So the main job is usually to configure the right knobs, not to build logging
from scratch.

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Training loop"
    desc="See how to customize LightningModule, trainer, or multi-stage training."
    icon="tabler:player-play"
    href="../finetuning/training-loop.html"
  />
  <DocCard
    title="Training Config"
    desc="See where `trainer`, `logger`, and `profiler` are configured."
    icon="tabler:settings-2"
    href="../../config/train_config.html"
  />
  <DocCard
    title="Trainer"
    desc="Read the detailed ESPnet3 Lightning trainer wrapper behavior."
    icon="tabler:layout-list"
    href="../../core/components/trainer.html"
  />
  <DocCard
    title="Parallel Config"
    desc="See how environment and worker settings are logged for cluster runs."
    icon="tabler:binary-tree-2"
    href="../../core/parallel/provider_runner.html"
  />
  <DocCard
    title="Provider / Runner"
    desc="Read the execution layer used by runner-based inference and helper workloads."
    icon="tabler:arrows-split-2"
    href="../../core/parallel/provider_runner.html"
  />
</DocCards>
