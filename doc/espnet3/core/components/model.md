---
title: ESPnet3 Model Configuration (Training)
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Model Configuration (Training)

This page explains how `model` and `task` in `train.yaml` map to model
construction for the `train` / `collect_stats` stages.

## Two modes: task (ESPnet2) vs model._target_ (custom)

**Use ESPnet2-style models (task)**

If you want to reuse an ESPnet2-derived model stack, set `task` and use an
ESPnet2-style `model:` block.

```yaml
task: espnet3.systems.esp2_asr.task.ASRTask
model:
  encoder: transformer
  decoder: transformer
  # ...ESPnet2-style config...
```

When `task` is set, `model:` is not instantiated by Hydra directly. Instead
`espnet3.utils.task_utils.get_espnet_model(task, model_config)` resolves the
ESPnet2 task class, fills in the task's default config, overlays your `model:`
block on top, and calls the task's own `build_model(...)`. This is the bridge
that lets ESPnet3 reuse an ESPnet2 task's model construction logic verbatim.
When `task` is omitted, `model:` must carry its own `_target_` and is
instantiated directly via Hydra instead.

Tip: you can start from existing ESPnet2 configs under `egs2/*/*/conf/*.yaml`.
See the [ESPnet2 task reference](#espnet2-task-reference) for task names and
links to the corresponding recipe docs.

Typical ASR `model:` keys in ESPnet2 configs:

| Key | Purpose |
| --- | --- |
| `encoder` / `encoder_conf` | Encoder type and settings. |
| `decoder` / `decoder_conf` | Decoder type and settings. |
| `model` / `model_conf` | ASR model head and loss settings (CTC/attention, etc.). |
| `frontend` / `frontend_conf` | Feature extraction (e.g., STFT/FBANK). |
| `specaug` / `specaug_conf` | SpecAugment settings. |
| `normalize` / `normalize_conf` | Feature normalization (e.g., global MVN). |

## ESPnet2 task reference

Below is a quick reference to ESPnet2 task names and their recipe docs.

| Task | Description |
| --- | --- |
| [`asr1`](../../../recipe/asr1.html) | Automatic Speech Recognition (Multi-tasking) |
| [`asr2`](../../../recipe/asr2.html) | Automatic Speech Recognition with Discrete Units |
| [`asvspoof1`](../../../recipe/asvspoof1.html) | Speaker Verification Spoofing and Countermeasures |
| [`cls1`](../../../recipe/cls1.html) | Classification |
| [`codec1`](../../../recipe/codec1.html) | Speech Codec |
| [`diar1`](../../../recipe/diar1.html) | Speaker Diarisation |
| [`enh1`](../../../recipe/enh1.html) | Speech Enhancement |
| [`enh_asr1`](../../../recipe/enh_asr1.html) | Speech Recognition with Speech Enhancement |
| [`enh_diar1`](../../../recipe/enh_diar1.html) | Speaker Diarisation with Speech Enhancement |
| [`enh_st1`](../../../recipe/enh_st1.html) | Speech-to-Text Translation with Speech Enhancement |
| [`hubert1`](../../../recipe/hubert1.html) | Self-supervised Learning |
| [`lid1`](../../../recipe/lid1.html) | Language Identification |
| [`lm1`](../../../recipe/lm1.html) | Language Modeling |
| [`mt1`](../../../recipe/mt1.html) | Machine Translation |
| [`s2st1`](../../../recipe/s2st1.html) | Speech-to-Speech Translation |
| [`s2t1`](../../../recipe/s2t1.html) | Weakly-supervised Learning (Speech-to-Text) |
| [`sds1`](../../../recipe/sds1.html) | ESPnet-SDS |
| [`slu1`](../../../recipe/slu1.html) | Spoken Language Understanding |
| [`speechlm1`](../../../recipe/speechlm1.html) | Speech Language Model |
| [`spk1`](../../../recipe/spk1.html) | Speaker Representation |
| [`ssl1`](../../../recipe/ssl1.html) | Self-supervised Learning |
| [`st1`](../../../recipe/st1.html) | Speech-to-Text Translation |
| [`svs1`](../../../recipe/svs1.html) | Singing Voice Synthesis |
| [`svs2`](../../../recipe/svs2.html) | ESPnet2 SVS2 Recipe TEMPLATE |
| [`tts1`](../../../recipe/tts1.html) | Text-to-Speech |
| [`tts2`](../../../recipe/tts2.html) | Text-to-Speech with Discrete Units |
| [`uasr1`](../../../recipe/uasr1.html) | Unsupervised Automatic Speech Recognition |

**Use custom/ESPnet3-only models (model._target_)**

If you want an ESPnet3-specific or fully custom model, implement it under your
recipe's `src/` directory and point `model._target_` to it:

```yaml
model:
  _target_: src.my_model.MyModel
  # custom args here
```

## Training-time forward contract

`ESPnetLightningModule` calls `model(**batch)` on every training and validation
step. Your model's `forward` **must** return exactly three values in this order:

```python
loss, stats, weight = model(**batch)
```

This is enforced — returning the wrong structure raises an error inside the
training loop.

### `loss` — scalar tensor for backprop

In the standard single-optimizer path, `loss` must be a scalar `torch.Tensor`
that requires grad. Lightning takes this return value and calls `.backward()`
on it automatically.

```python
loss = compute_ctc_loss(...)   # scalar, requires_grad=True
```

### `stats` — the logging dict

`stats` must be a `dict[str, Tensor]`. The training loop logs every entry
directly to TensorBoard / W&B under the key `{mode}/{key}`, where `mode` is
`train` or `valid`.

```python
stats = {
    "loss": loss.detach(),   # → logged as "train/loss" / "valid/loss"
    "acc":  acc.detach(),    # → logged as "train/acc"  / "valid/acc"
}
```

**Every value must be detached.** Forgetting `.detach()` keeps the gradient
graph alive across steps and wastes GPU memory. It does not raise an error, so
the mistake is easy to miss.

### `weight` — batch size for weighted averaging

`weight` is a scalar tensor representing the number of samples in the batch.
The training loop passes it as `batch_size` to Lightning's `log_dict`, which
enables correct weighted averaging across variable-length batches and DDP
workers.

```python
weight = speech.new_tensor(speech.shape[0])   # same device as loss
```

Use `tensor.new_tensor(value)` rather than `torch.tensor(value,
device=loss.device)` to match the device of an existing tensor automatically.

### Full example

```python
class MyASRModel(torch.nn.Module):
    def forward(self, speech, speech_lengths, text, text_lengths, **kwargs):
        loss, acc = self._compute_loss(
            speech, speech_lengths, text, text_lengths
        )
        stats = {
            "loss": loss.detach(),
            "acc":  acc.detach(),
        }
        weight = speech.new_tensor(speech.shape[0])
        return loss, stats, weight
```

### NaN / Inf handling

If `loss` is NaN or Inf the **entire batch is skipped across all DDP workers**
at once. After 100 consecutive NaN batches the training loop raises
`RuntimeError` and stops.

### Multi-optimizer (GAN-style) training

For GAN-style or other multi-optimizer training, `loss` is replaced by one or
more `OptimizationStep` objects that route each loss to the correct optimizer.
`stats` and `weight` keep the same structure; `weight` may be `None`.

```python
from espnet3.components.modeling.optimization_spec import OptimizationStep

def forward(self, **batch):
    g_loss = ...
    d_loss = ...
    stats = {
        "generator_loss":     g_loss.detach(),
        "discriminator_loss": d_loss.detach(),
    }
    return [
        OptimizationStep(loss=g_loss, name="generator"),
        OptimizationStep(loss=d_loss, name="discriminator"),
    ], stats, None
```

The training loop automatically logs `train/generator/loss`,
`train/discriminator/loss`, and per-optimizer update steps in addition to the
keys in `stats`. Only the optimizers named in the returned steps are updated for
that batch; others are left untouched.

See [Optimizer Configuration](./optimizer_configuration.html) for YAML wiring
and per-optimizer gradient clipping.

## Collect-stats support (collect_feats)

If you want to use `collect_stats`, your model should implement `collect_feats()`.
See:

- [Collect-stats stage](../../stages/collect-stats.html)
- [Training configuration](../../config/train_config.html)

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Trainer"
    desc="See how model outputs are consumed by the training wrapper."
    icon="tabler:player-play"
    href="./trainer.html"
  />
  <DocCard
    title="Metrics"
    desc="See how model outputs later flow into metrics and evaluation."
    icon="tabler:gauge"
    href="./metrics.html"
  />
  <DocCard
    title="Training configuration"
    desc="See where model selection and normalization live in YAML."
    icon="tabler:settings-2"
    href="../../config/train_config.html"
  />
</DocCards>
