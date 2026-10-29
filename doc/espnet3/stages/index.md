# Stages

ESPnet3 recipes run named stages, invoked with `--stages <name> [<name> ...]` (or `--stages all`)
on `run.py`. A stage is a plain method on the recipe's `System` class
([`BaseSystem`](https://github.com/espnet/espnet/blob/master/espnet3/systems/base/system.py),
[`ASRSystem`](https://github.com/espnet/espnet/blob/master/espnet3/systems/esp2_asr/system.py),
[`F5TTSSystem`](https://github.com/espnet/espnet/blob/master/espnet3/systems/f5tts/system.py)); the CLI
only selects which declared stage methods to call, in that system's order.

## Stage declarations and launch

The source of stage names, ordering, configuration roles, and log directories is
`System.stages`, a tuple of `StageSpec` declarations. It is not a constant in
`run.py`. A recipe entry point delegates to the shared launcher:

```python
from espnet3.systems.esp2_asr.system import ASRSystem
from espnet3.systems.launch import launch

if __name__ == "__main__":
    launch(ASRSystem)
```

`launch` derives config flags from the roles in those declarations. For example,
`config="training"` gives `--training_config`; `config=("training", "publication")`
uses both roles. Stage methods take no runtime arguments: read their settings
from `self.stage_configs["stage_name"]`.

| Stage | Role(s) | Log directory key | Guide |
| --- | --- | --- | --- |
| `create_dataset` | `training` | `data_dir` | [Dataset preparation](./create-dataset.html) |
| `collect_stats` | `training` | `stats_dir` | [Statistics](./collect-stats.html) |
| `train` | `training` | `exp_dir` | [Training](./train.html) |
| `infer` | `inference` | `inference_dir` | [Inference](./inference.html) |
| `measure` | `metrics` | `inference_dir` | [Metrics](./metrics.html) |
| `pack_model` | `training`, `publication` | experiment fallback | [Publication](./publish.html) |
| `upload_model` | `publication` | experiment fallback | [Publication](./publish.html) |
| `pack_demo`, `upload_demo` | `demo` | `pack.out_dir` | [Demo](./demo.html) |

`ASRSystem` inserts `train_tokenizer` after `create_dataset`, with role `training`
and log directory `tokenizer.save_path`. `F5TTSSystem` inserts
`remove_long_short` and then `create_token_list` there instead, with log paths
`remove_long_short.save_path` and `create_token_list.save_path`. There is no
`TTSSystem` class in the current implementation.

`--stages all`, or omitting `--stages`, selects every declared stage. An explicit
subset runs in declaration order, regardless of CLI order. Selecting a stage does
not execute its prerequisites automatically. Supply every requested config role,
or use `--exp_dir` to enable saved-config lookup. `--dry_run` resolves configs and
logs the plan without executing stage methods; construction may still create the
experiment directory.

::: important Requested stage configurations
The current CLI retains only the requested stage entries on `system.stage_configs`.
ASR `train` also reads the `train_tokenizer` entry, and `measure` reads `infer`.
Request these together (`--stages train_tokenizer train`, or `infer measure`) with
their configs. Model packing also needs predecessor entries; see the direct
system example in [publication](./publish.html) to pack without rerunning training.
Passing a role flag alone does not retain its stage entry.
:::

The constructor `BaseSystem(configs=..., stages_to_run=...)` builds independent
config objects for the prefix through the requested stages. Inherited settings
are replaced by whole top-level keys, not recursively merged across stages.
`run_stages` bakes own keys under `${exp_dir}/config/<stage>.yaml` and writes a full
snapshot to `<stage-log-dir>/<stage>.yaml`; these are different records.

## How to use this overview

Use the cards below to choose the stage you need. Each stage guide explains its
purpose, required configuration, inputs and outputs, implementation API, safe
re-run behavior, and the next stage to run. The execution implementation is
available through [`BaseSystem`](https://github.com/espnet/espnet/blob/master/espnet3/systems/base/system.py),
[`ASRSystem`](https://github.com/espnet/espnet/blob/master/espnet3/systems/esp2_asr/system.py),
[`F5TTSSystem`](https://github.com/espnet/espnet/blob/master/espnet3/systems/f5tts/system.py), and
[`run_stages`](https://github.com/espnet/espnet/blob/master/espnet3/utils/stages_utils.py).

For a normal ASR workflow, start with [dataset preparation](./create-dataset.html),
then [statistics](./collect-stats.html), [training](./train.html),
[inference](./inference.html), [measurement](./metrics.html), and finally
[publication](./publish.html) or [demo packaging](./demo.html). See
[configuration files](../config/index.html) before running a stage and the
[generated Python API](../api-reference.html) for implementation details.

::: warning No rank guard on multi-GPU local launches
`run_stages()`
([`espnet3/utils/stages_utils.py`](https://github.com/espnet/espnet/blob/master/espnet3/utils/stages_utils.py))
only special-cases the `train` stage's log file naming per rank (`rank0` vs. `per_rank` in
`stage_log_mode`); it does not gate any stage on process rank. If a multi-GPU `training.yaml`
(`devices > 1` with Lightning's default local subprocess launcher, not `torchrun`/`srun`) requests
multiple stages (e.g. the default `--stages all`), Lightning re-executes `run.py` once per local
rank; after `trainer.fit()` returns in each rank's process, every rank continues on to run the
remaining stages (`infer`, `measure`, `pack_model`, `upload_model`, ...) concurrently against the
same output directories. Launch multi-GPU jobs with a train-only invocation, and run preparation and post-training stages in separate
single-process invocations. External launchers start a script on each task too;
they do not supply a missing stage-body rank guard.
:::

## Stage reference

<DocCards>
  <DocCard
    title="create_dataset"
    desc="Download or build datasets for your recipe."
    icon="tabler:database"
    href="./create-dataset.html"
  />
  <DocCard
    title="collect_stats"
    desc="Compute feature shapes and global statistics."
    icon="tabler:chart-bar"
    href="./collect-stats.html"
  />
  <DocCard
    title="train"
    desc="Run Lightning training with training.yaml."
    icon="tabler:school"
    href="./train.html"
  />
  <DocCard
    title="infer"
    desc="Write hypothesis outputs under inference_dir."
    icon="tabler:bolt"
    href="./inference.html"
  />
  <DocCard
    title="measure"
    desc="Compute metrics (WER, MOS, SI-SDR, …) from inference outputs."
    icon="tabler:ruler-measure"
    href="./metrics.html"
  />
  <DocCard
    title="pack_model / upload_model"
    desc="Bundle and publish a trained model to HuggingFace Hub."
    icon="tabler:package-export"
    href="./publish.html"
  />
  <DocCard
    title="pack_demo / upload_demo"
    desc="Generate and upload a Gradio demo UI."
    icon="tabler:device-desktop"
    href="./demo.html"
  />
</DocCards>

## Related pages

<DocCards>
  <DocCard
    title="What is a recipe"
    desc="How run.py, BaseSystem, and named stages fit together."
    icon="tabler:puzzle"
    href="../get-started/what-is-a-recipe.html"
  />
  <DocCard
    title="Adding a stage"
    desc="See the contributor guide for extending a System class and wiring stages into run.py."
    icon="tabler:tool"
    href="../contributing/adding-a-stage.html"
  />
</DocCards>
