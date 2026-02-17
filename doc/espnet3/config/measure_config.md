---
title: ESPnet3 Measure Configuration
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Measure Configuration

This page explains the `measure.yaml` schema used by the measure stage. It
consumes inference outputs from `infer_dir` and reports metric results.

## Minimum required keys (typical measurement run)

Required:

- `infer_dir`
- `metrics` (at least one entry)
- `dataset.test` (used to enumerate test names)

Common optional:

- `recipe_dir`, `exp_dir`, `dataset_dir` (path scaffold)

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

metrics:
  - metric:
      _target_: my_pkg.metrics.MyMetric
    inputs:
      ref: ref
      hyp: hyp
```

## ✅ Config sections overview

| Section | Description |
| --- | --- |
| `recipe_dir`, `exp_dir`, `infer_dir`, ... | Path scaffold for outputs and metric result files. |
| `dataset` | Test set definitions used to enumerate test names. |
| `metrics` | Metric classes and input mappings. |

## Core config layout (ASR example)

```yaml
recipe_dir: .
exp_tag: asr_template_eval
exp_dir: ${recipe_dir}/exp/${exp_tag}
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

metrics:
  - metric:
      _target_: espnet3.systems.asr.metrics.wer.WER
      clean_types: ["whisper_basic"]
    inputs:
      ref: ref
      hyp: hyp
```

## Metrics and inputs

Each entry in `metrics` provides a `metric` block (Hydra instantiation) and
optional `inputs`. If `inputs` is omitted, ESPnet3 falls back to the metric's
`ref_key` and `hyp_key` attributes.

Inputs map to SCP files under `${infer_dir}/${test_name}`. For example,
`inputs.ref: ref` expects `ref.scp` and `inputs.hyp: hyp` expects `hyp.scp`.

The metric list is evaluated one by one, so each list entry produces its own
results block in `measures.json`.

## Output directory layout

During inference, ESPnet3 writes SCP outputs under a per-test-set folder:

```
${infer_dir}/
  test-clean/
    ref.scp
    hyp.scp
  test-other/
    ref.scp
    hyp.scp
```

Each `inputs` entry maps to a file in that test directory:

- `inputs.ref: ref` -> `${infer_dir}/${test_name}/ref.scp`
- `inputs.hyp: hyp` -> `${infer_dir}/${test_name}/hyp.scp`

The loaded inputs are passed to the metric class as a dict, so your custom
metrics can consume any field you register in `inputs`.

If you want to write custom metrics, see [Custom metrics](../core/components/metrics.md).
