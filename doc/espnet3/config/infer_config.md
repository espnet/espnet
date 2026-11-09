---
title: ESPnet3 Inference Configuration
author:
  name: "Masao Someki"
date: 2026-04-15
---

# ESPnet3 Inference Configuration

## Minimum required keys

Typical inference runs need:

- `dataset.test`
- `model`
- `provider`
- `runner`
- `input_key`
- `inference_dir`

Optional:

- `contract`
- `output_fn`
- `output_keys`
- `idx_key`
- `batch_size`
- `parallel`

## Config sections overview

| Section | Description |
| --- | --- |
| `recipe_dir`, `exp_dir`, `inference_dir`, ... | path scaffold for inference outputs |
| `dataset` | test-set definitions resolved through `DataOrganizer` |
| `model` | inference-time model entrypoint and arguments |
| `input_key` | which dataset field or fields are passed into the model |
| `contract` | declares model inputs and outputs with their kinds, sample rates, channels, and normalizers (recommended) |
| `output_fn` | import path to the formatting function used by `InferenceRunner` (legacy; cannot be combined with `contract`) |
| `output_keys`, `idx_key` | controls SCP output names and utterance ID key |
| `batch_size` | enables batched runner execution |
| `parallel` | local, multi-worker, or cluster execution settings |

## Default values

| Key | Default value |
| --- | --- |
| `recipe_dir` | `.` |
| `data_dir` | `${recipe_dir}/data` |
| `exp_tag` | empty |
| `exp_dir` | `${recipe_dir}/exp/${exp_tag}` |
| `inference_dir` | `${exp_dir}/${self_name:}` |
| `parallel.env` | `local` |
| `parallel.n_workers` | `1` |
| `input_key` | `speech` |
| `contract` | unset |
| `output_fn` | unset (model output must already be a dict, or a list of dicts in batched mode) |
| `provider._target_` | `espnet3.systems.base.inference_provider.InferenceProvider` |
| `runner._target_` | `espnet3.systems.base.inference_runner.InferenceRunner` |

## Naming behavior

If `training_config` is provided alongside `inference.yaml`, inference will inherit the values for `exp_tag` and `exp_dir` from training. If these keys are also defined in `inference.yaml`, they will be overwritten.

If inference runs standalone, `inference.yaml` must carry its own experiment
identity, typically through:

```yaml
exp_tag: my_eval
exp_dir: ${recipe_dir}/exp/${exp_tag}
```

## Minimal example

```yaml
exp_tag: inference_beam5

dataset:
  test:
    - name: test
      data_src: mini_an4/asr
      data_src_args:
        split: test

model:
  _target_: espnet2.bin.asr_inference.Speech2Text
  asr_train_config: ${exp_dir}/config.yaml
  asr_model_file: ${exp_dir}/last.ckpt

# Use contract instead of output_fn
contract:
  inputs:
    speech:
      kind: audio
      label: Input Audio
      sample_rate: 16000
      rate_policy: resample
      channels: 1
  outputs:
    text:
      kind: text
      label: Transcription
```

## Model definition

`model` should point to an inference-time callable. For ESPnet2-compatible
recipes this is often an `espnet2.bin.*` helper such as
`espnet2.bin.asr_inference.Speech2Text`.

When you use a custom inference stack, provide your own `_target_` and
arguments. The instantiated object should accept the inputs named by
`input_key`. With a contract, the runner validates the model's output against
the contract's output fields; without a contract, `output_fn` interprets the
return value.

See [Model](../core/components/model.html) for model implementation details.

## Output directory layout

Inference writes SCP outputs under `inference_dir`, one folder per test-set
name:

```text
${inference_dir}/
  test-clean/
    hyp.scp
    wav.scp
  test-other/
    hyp.scp
```

The exact filenames are determined by:

- `output_keys` if it is set
- otherwise the keys returned by `output_fn` for the first sample, excluding
  `idx_key`
- or the contract `outputs` names if a contract is used

## Contract

The `contract` section in `inference.yaml` declares the model's inputs and
outputs with their types, constraints, and optional normalizations. This is the
recommended way to describe model behavior; it replaces the legacy `output_fn`
approach and provides built-in validation.

A minimal contract for audio-to-text ASR looks like:

```yaml
contract:
  imports:
    - src.contracts  # optional: load custom kinds/normalizers from a recipe module
  inputs:
    speech:
      kind: audio
      label: Input Audio
      sample_rate: 16000
      rate_policy: resample  # resample mismatched rates; use `reject` to refuse them
      channels: 1           # require exactly 1 channel; remove to pass multichannel through
      # normalize: [mean_channels]  # optional: apply normalization before model sees input
  outputs:
    text:
      kind: text
      label: Transcription
      # normalize: [strip, to_lower]  # optional: apply text normalization
```

### Contract fields

- `inputs` and `outputs` are mappings from field names to field descriptors.
- Each descriptor must include a `kind` (e.g., `audio`, `text`, `number`,
  `segments`).
- Optional fields are marked with `optional: true`.
- Audio fields can specify `sample_rate`, `rate_policy` (`resample` or
  `reject`), `channels` (exact channel count required), and `normalize`
  (list of normalizers like `mean_channels`, `first_channel`).
- Text and other kinds accept `normalize` with normalizers appropriate to that
  kind.
- Unknown keys in the contract raise a `ValueError` during loading.

### Built-in kinds

- `audio`: An audio signal with optional sample rate and channel information.
  - Options: `sample_rate`, `rate_policy`, `channels`, `normalize`
- `text`: A text string.
  - Options: `normalize`
- `number`: A numeric value.
- `segments`: A list of segments with start/end times.

### Normalizers

Normalizers are functions that transform values of a specific kind. Built-in
normalizers include:

- Audio: `mean_channels`, `first_channel`
- Text: `strip`, `to_lower`

Custom kinds and normalizers can be registered via `register_kind` and
`register_normalizer` from `espnet3.contract`, then loaded through the
`imports` list. Loading custom code from a published bundle requires
`trust_user_code: true` in the model configuration.

### Input/output matching

- `input_key` must match the names declared in `contract.inputs` exactly; no
duplicates are allowed.
- Optional inputs are declared with `optional: true` in the contract and must
also be present in `input_key`. Within one batch, all samples must either all
have or all lack an optional input, otherwise a `KeyError` is raised.
- The model output must match the contract's output field names; the runner
validates this and produces SCP files named after the output fields.

### Audio behavior

When an audio input has a known sample rate, it is automatically resampled to
the declared `sample_rate` (using `rate_policy: resample`). The model receives
an `Audio` object with all channels preserved unless `channels: N` is set,
which validates the exact channel count. For stereo input with a mono model,
add `normalize: [mean_channels]` or `[first_channel]` to the input descriptor.

### Legacy and exclusions

- `output_fn` cannot be combined with a contract; setting both raises
  `ValueError("output_fn cannot be combined with an inference contract")`.
- The ESPnet2 ASR system moved n-best text formatting from `output_fn` into the
  runner, so the ASR template no longer sets `output_fn`.
- The contract travels with the packed model in `inference.yaml`, so
  `InferenceModel.from_pretrained` and `from_packed` apply it automatically.

## Output control

Useful keys:

- `idx_key`: sample ID key, default `utt_id`
- `output_keys`: explicit SCP filenames to write
- `output_artifacts`: artifact-writing config for non-scalar outputs

Built-in artifact behavior:

- `dict` -> JSON
- `numpy.ndarray` -> NPY
- CPU `torch.Tensor` -> NPY
- unknown Python object -> pickle

Supported artifact types:

| Type | Saved as | Notes |
| --- | --- | --- |
| `wav` | `.wav` | requires `sample_rate` |
| `npy` | `.npy` | good for NumPy arrays or CPU tensors |
| `json` | `.json` | good for `dict` outputs |
| `pickle` | `.pkl` | fallback for custom Python objects |
| `writer` | custom path | uses a user-defined function via `_target_` |

See [Inference stage](../stages/inference.html) for:

- WAV output example
- custom writer example
- artifact directory structure

## Dataset naming

`dataset.test[*].name` becomes:

- the selected test-set key
- the subdirectory name under `inference_dir`

This same test-set name is later reused by `measure()`.

## Batched execution

If `batch_size` is set, `InferenceRunner.forward()` receives a list of indices
and passes list-valued inputs to the model.

In that case:

- `data` becomes a list of dataset samples
- `idx` becomes a list of indices
- `output_fn` must return a list of output dicts

If your model or `output_fn` does not support batched list inputs, leave
`batch_size` unset or `null`.

Minimal batched example:

```python
def build_output(*, data, model_output, idx):
    return [
        {
            "utt_id": item["uttid"],
            "hyp": hyp,
            "ref": item.get("text", ""),
        }
        for item, hyp in zip(data, model_output["text"])
    ]
```

This is why the document examples keep `output_fn` small: most recipe-specific
formatting problems are easier to solve there than by replacing the whole stage.

## Parallel execution

Minimal local example:

```yaml
parallel:
  env: local
  n_workers: 1
```

Minimal SLURM example:

```yaml
parallel:
  env: slurm
  n_workers: 8
  options:
    queue: gpu
    cores: 8
    processes: 1
    memory: 16GB
    walltime: 30:00
    job_extra_directives:
      - "--gres=gpu:1"
```

Parallel execution details are documented here:

- [Provider / Runner](../core/parallel/provider_runner.html)
- [Multi-GPU / multi-node](../guides/scaling/multi-node.html)

## Related pages

- [Inference stage](../stages/inference.html)
- [Dataset references and builders](../core/components/data-organizer.html)
- [Training configuration](./train_config.html) — see its Resolvers section for `${self_name:}` and friends
