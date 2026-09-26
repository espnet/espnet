---
title: ESPnet3 Train Stage
author:
- name: "Masao Someki"
- name: "Elias Naske"
date: 2026-05-15
---

# ESPnet3 Train Stage

The `train` stage runs model training using a [PyTorch Lightning trainer](../core/components/trainer.html) based on the dataset and hyperparameters defined in `training.yaml` and saves model checkpoints and logs. For ASR recipes, it also trains the tokenizer first if one is not already cached.

## 1. Run

```bash
python run.py --stages train --training_config conf/training.yaml
```

`--training_config` is required for `train` (and for `create_dataset`, `train_tokenizer`, and `collect_stats`) -- it is the config `run.py` uses to resolve `exp_dir`/`exp_tag`. Omitting `--stages` runs the recipe's full default stage list (`create_dataset -> train_tokenizer -> collect_stats -> train -> infer -> measure -> ...`) in a single process; see the "Multi-GPU training" and "Normalization and `collect_stats`" sections below for why that matters.

## 2. Configuration

Training is configured in `training.yaml` using the sections shown in the table below.
For a detailed list of options, see [Training Configuration](../config/train_config.html) and the links in the table.


| Section                    | Description                                 | Details                                                                                    |
| -------------------------- | ------------------------------------------- | ------------------------------------------------------------------------------------------ |
| `task`                     | task entrypoint for ESPnet2-style models; requires `model` to also be set |
| `model`                    | model definition and normalization settings |
| `dataset`                  | `train` and `valid` splits                  | [Data Organizer](../core/components/data-organizer.html)                                     |
| `dataloader`               | collate and iterator settings               | [Dataloader + Collate](../core/components/dataloader.html)                                   |
| `trainer`                  | Lightning trainer configuration             | [Trainer](../core/components/trainer.html)                                                   |
| `optimizer`, `scheduler`   | single-optimizer training path              | [Optimizer + Scheduler](../core/components/optimizer_configuration.html)                     |
| `optimizers`, `schedulers` | named multi-optimizer path                  | [Multiple Optimizers and Schedulers](../core/components/multiple_optimizers_schedulers.html) |
| `exp_dir`                  | training output directory                   |
| `tokenizer`                | ASR only -- SentencePiece training settings, see below |

::: warning `model` must be set when `task` is set
`save_espnet_config` ([`espnet3/utils/task_utils.py`](https://github.com/espnet/espnet/blob/master/espnet3/utils/task_utils.py))
pops `model` off the resolved config and checks it for a `_target_` key without first checking it is
non-`None`. Leaving `model:` blank while `task:` is set (e.g. a `TEMPLATE`-style placeholder that was
never filled in) fails with a bare `TypeError: argument of type 'NoneType' is not iterable` instead of
a message naming `model` as the missing field.
:::

## 3. ASR tokenizer training

`ASRSystem.train()` checks `tokenizer.save_path` for an existing
`<model_type>.model`/`<model_type>.vocab` pair; if both files are present it
skips straight to training, otherwise it runs `train_tokenizer` first, which
gathers text via `tokenizer.text_builder.func` and trains SentencePiece with
`tokenizer.vocab_size` and `tokenizer.model_type`.

::: warning
`train_tokenizer` only forwards `vocab_size` and `model_type` to SentencePiece
training. `tokenizer.character_coverage` and `tokenizer.user_defined_symbols`
are accepted by the underlying training helper but are not read from
`training_config.tokenizer`, so setting either in `training.yaml` currently has
no effect.
:::

The trained `<model_type>.model`/`<model_type>.vocab` and a derived
`tokens.txt` are written under `tokenizer.save_path`. Recipes wire these into
the model config via plain OmegaConf interpolation, not via any code in
`ASRSystem`:

```yaml
tokenizer:
  save_path: ${data_dir}/${tokenizer.model_type}_${tokenizer.vocab_size}

model:
  token_list: ${tokenizer.save_path}/tokens.txt
```

## 4. Normalization and `collect_stats`

The base `collect_stats()` removes `model.normalize`/`model.normalize_conf`
from the config before building the model, so stats can be collected before
those statistics exist.

::: warning
This removal happens **in place** on the shared `training_config` object, not
on a copy. Because the default stage list runs `collect_stats` and `train` on
the same `System` instance in one process, a subsequent `train()` call in that
same run builds the model from the same mutated config -- so
`model.normalize`/`model.normalize_conf` are silently missing during training
even when they are set in `training.yaml`. Until this is fixed, run
`collect_stats` and `train` as separate `python run.py` invocations so each
one loads its own fresh config:

```bash
python run.py --stages collect_stats --training_config conf/training.yaml
python run.py --stages train         --training_config conf/training.yaml
```

`TTSSystem` overrides `collect_stats` to avoid this, so it only affects
`BaseSystem`/`ASRSystem`.
:::

## 5. Outputs

Training outputs are written under `exp_dir` (by default
`${recipe_dir}/exp/${exp_tag}`, where `exp_tag` defaults to the stem of the
`--training_config` file via the `${self_name:}` resolver), including:

- Checkpoints (selected by `best_model_criterion`)
- Logs (per-stage log files under `exp_dir`)
- (If configured) TensorBoard output under `${exp_dir}/tensorboard`
- `config.yaml`, the resolved model config, when `training_config.task` is set

## 6. Resuming and re-running

Re-running `train` does not skip or resume automatically -- it re-instantiates
the trainer and calls `trainer.fit()` again. Whether that continues from a
checkpoint or restarts is controlled by `training_config.fit`, forwarded as
`trainer.fit(**fit)` (e.g. set `fit: {ckpt_path: ...}` to resume). The one part
of `train` that is idempotent by default is tokenizer training: it is skipped
whenever `tokenizer.save_path` already has a matching model/vocab pair.

## 7. Multi-GPU training

Setting `trainer.devices` above 1 (with the default `strategy: auto`, which
resolves to DDP) trains across multiple local processes.

::: warning
`run_stages()` has no rank guard for any stage besides log-file naming for
`train` itself. When Lightning's local (non-`torchrun`) subprocess launcher
spawns one child process per additional GPU, each child re-executes `run.py`
with the same argv from the top -- so every stage in that invocation's
`--stages` list other than `train` (e.g. `infer`, `measure`, publication
stages, or the full `DEFAULT_STAGES` default) also re-runs once per rank.
Until this is fixed, launch a multi-GPU run with `--stages train` on its own,
and run `infer`/`measure`/publication stages afterwards as separate
single-process invocations.
:::

## Related pages

**Stage API:** [`train`](../../guide/espnet3/systems/train.html),
[`ASRSystem`](../../guide/espnet3/systems/ASRSystem.html),
[`ESPnetLightningModule`](../../guide/espnet3/components/ESPnetLightningModule.html), and
[`Trainer`](../core/components/trainer.html). Continue with
[inference](./inference.html) after checkpoints are available.
<DocCards :cols="3">
  <DocCard
    title="Training configuration"
    desc="See all options for cofiguring the train stage."
    icon="tabler:file-code"
    href="../config/train_config.html"
  />
  <DocCard
    title="Inference stage"
    desc="Information on the inference stage."
    icon="tabler:puzzle"
    href="./inference.html"
  />
  <DocCard
    title="Trainer"
    desc="Information about the trainer component"
    icon="tabler:tool"
    href="../core/components/trainer.html"
  />
</DocCards>
