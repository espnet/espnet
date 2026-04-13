---
title: ESPnet3 Inference Configuration
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Inference Configuration

This page explains the `infer.yaml` schema used by the infer stage.
The example below is from an ASR recipe, so it uses ASR-style dataset
definitions.

## Minimum required keys (typical inference run)

Required:

- `model` (Hydra target; instantiated with a `device` argument)
- `dataset.test` (at least one test set; names are used as output subfolders)
- `infer_dir` (root output directory for `.scp` files)
- `input_key` (which dataset field(s) to pass into the model)
- `output_fn` (import path to a function that formats runner outputs)

Common optional:

- `parallel` (defaults to local if omitted)
- `exp_dir`, `recipe_dir`, `dataset_dir` (path scaffold)
- `output_keys` (explicit list of SCP keys to write; otherwise inferred)
- `idx_key` (default: `uttid`)
- `batch_size` (enables batched runner execution)

Minimal example:

```yaml
infer_dir: exp/my_exp/infer

dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  test:
    - name: test
      dataset:
        _target_: src.dataset.MyDataset
        data_dir: /path/to/data

model:
  _target_: src.infer.MyInferenceModel

input_key: speech
output_fn: src.infer.output_fn
```

## ✅ Config sections overview

| Section | Description |
| --- | --- |
| `recipe_dir`, `exp_dir`, `infer_dir`, ... | Path scaffold for outputs and inference results. |
| `dataset` | Test set definitions. |
| `model` | Inference model entrypoint and parameters (instantiated with `device=`). |
| `input_key` | Dataset field name (or list of names) passed into the model. |
| `output_fn` | Import path to a result formatting function. |
| `output_keys`, `idx_key` | Controls SCP output filenames and the ID field name. |
| `parallel` | Dask / cluster settings for runners. |

## Core config layout (ASR example)

```yaml
recipe_dir: .
exp_tag: asr_template_eval
exp_dir: ${recipe_dir}/exp/${exp_tag}
stats_dir: ${recipe_dir}/exp/stats
infer_dir: ${exp_dir}/infer
dataset_dir: /path/to/your/dataset

dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  test:
    - name: test-clean
      dataset:
        _target_: src.dataset.LibriSpeechDataset
        data_dir: ${dataset_dir}
        split: test-clean

model:
  _target_: espnet2.bin.asr_inference.Speech2Text
  asr_train_config: ${exp_dir}/config.yaml
  asr_model_file: ${exp_dir}/last.ckpt

parallel:
  env: local
  n_workers: 1

input_key: speech
output_fn: src.infer.output_fn

```

## Model definition

`model` should point to an inference-time callable. For ESPnet2-compatible
recipes this is typically an `espnet2.bin.*` inference helper such as
`espnet2.bin.asr_inference.Speech2Text`. When using a custom inference stack,
provide your own `_target_` and arguments.

See [Inference providers](../core/parallel/inference_provider.md) for the provider/runner
contract and model construction details.

## Output directory layout

Inference writes `.scp` outputs under `infer_dir`, one folder per test set name:

```
${infer_dir}/
  test-clean/
    hyp.scp
    hyp0.scp
    hyp1.scp
  test-other/
    hyp.scp
```

The exact filenames are determined by:

- `output_keys` (if set), or
- the keys returned by `output_fn` (excluding `idx_key`) for the first sample.

## Batched execution (`batch_size`)

If `batch_size` is set, ESPnet3 may call `InferenceRunner.batch_forward()`.
When the model implements `batch_forward`, it is preferred and receives lists
of inputs (one list per `input_key`).

In that case, `output_fn` may be called with:

- `data`: a list of dataset items
- `idx`: a list of indices
- `model_output`: your model's batched output structure

If you don't want to handle batched `output_fn`, leave `batch_size` unset (or
avoid providing `batch_forward` on the model).

## Parallel execution

Parallel execution is documented separately:

- [Provider / Runner](../core/parallel/provider_runner.md)
- [Multi-GPU / multi-node](../core/parallel/multiple_gpu.md)
