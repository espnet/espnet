# Stages

ESPnet3 recipes run named stages, invoked with `--stages <name> [<name> ...]` (or `--stages all`)
on `run.py`. A stage is a plain method on the recipe's `System` class
([`BaseSystem`](https://github.com/espnet/espnet/blob/master/espnet3/systems/base/system.py),
[`ASRSystem`](https://github.com/espnet/espnet/blob/master/espnet3/systems/asr/system.py),
[`TTSSystem`](https://github.com/espnet/espnet/blob/master/espnet3/systems/tts/system.py)); the CLI
only selects which stage methods to call, in a fixed canonical order.

## Canonical stage order

Whatever order stage names are passed on the CLI, they always run in this order (from
`DEFAULT_STAGES` in
[`egs3/TEMPLATE/asr/run.py`](https://github.com/espnet/espnet/blob/master/egs3/TEMPLATE/asr/run.py)):

| # | Stage | System | Page |
|---|---|---|---|
| 1 | `create_dataset` | all | [create-dataset.html](./create-dataset.html) |
| 2 | `train_tokenizer` | ASR | *(see [ASRSystem](https://github.com/espnet/espnet/blob/master/espnet3/systems/asr/system.py))* |
| — | `create_token_list`, `remove_long_short` | TTS only, before `train` | *(see [TTSSystem](https://github.com/espnet/espnet/blob/master/espnet3/systems/tts/system.py))* |
| 3 | `collect_stats` | all | [collect-stats.html](./collect-stats.html) |
| 4 | `train` | all | [train.html](./train.html) |
| 5 | `infer` | all | [inference.html](./inference.html) |
| 6 | `measure` | all | [metrics.html](./metrics.html) |
| 7 | `pack_model` / `upload_model` | all | [publish.html](./publish.html) |
| 8 | `pack_demo` / `upload_demo` | all | [demo.html](./demo.html) |

TTS recipes additionally run `create_token_list` and `remove_long_short` ahead of `train`; these are
`TTSSystem`-only stages with no shipped TTS recipe yet, so they are not documented as standalone
pages here. `--stages all` expands to every stage a recipe's `run.py`
declares (`resolve_stages` in
[`espnet3/utils/stages_utils.py`](https://github.com/espnet/espnet/blob/master/espnet3/utils/stages_utils.py));
passing an explicit subset (e.g. `--stages infer measure`) still executes in the table order above,
not CLI order.

## How to use this overview

Use the cards below to choose the stage you need. Each stage guide explains its
purpose, required configuration, inputs and outputs, implementation API, safe
re-run behavior, and the next stage to run. The execution implementation is
available through [`BaseSystem`](../../guide/espnet3/systems/BaseSystem.html),
[`ASRSystem`](../../guide/espnet3/systems/ASRSystem.html),
[`TTSSystem`](../../guide/espnet3/systems/TTSSystem.html), and
[`run_stages`](../../guide/espnet3/utils/run_stages.html).

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
same output directories. Launch multi-GPU jobs with `torchrun`/`srun` (which does not re-exec
`run.py` per rank), or run the post-`train` stages as a separate single-process `run.py` invocation
after training finishes.
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
