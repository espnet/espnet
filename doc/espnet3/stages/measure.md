---
title: 📘 ESPnet3 Measurement Stage
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Measurement Stage

This document explains the **measurement stage** in ESPnet3, implemented in:

* `espnet3.systems.base.measure.measure`
* `espnet3.components.metrics.abs_metric.AbsMetrics`

Measurement reads the `ref.scp` and `hyp.scp` files produced by inference and
writes a `measures.json` summary.

For the full metric interface (how `AbsMetrics` is called, how SCPs are aligned,
and how to implement custom metrics), see:

- [ESPnet3 Metrics](../core/components/metrics.md)

## Quick usage

### Run

```bash
python run.py --stages measure --measure_config conf/measure.yaml
```

### Configure (in `measure.yaml`)

Keep the core settings in `measure.yaml`. For the full list, see
[Measurement configuration](../config/measure_config.md).

| Config section | Description |
| -------------- | ----------- |
| `dataset` | Dataset organizer and test splits. Measurement uses this to iterate test set names. |
| `metrics` | List of metric definitions. Each entry specifies `metric` and optional `inputs`. |
| `infer_dir` | Location of `.scp` files under `infer_dir/<test_name>/`. |

<!-- TODO(masao): update this section after the PR that removes the dataset dependency in measure(). -->

### Outputs

Measurement writes:

```text
<infer_dir>/measures.json
```

## Developer Notes

### 🧩 Config fields used during measurement

A minimal `measure_config` for measurement looks like:

```yaml
infer_dir: exp/asr_example/infer

dataset:
  _target_: espnet3.components.data.data_organizer.DataOrganizer
  test:
    - name: test-clean
      dataset:
        _target_: ...
    - name: test-other
      dataset:
        _target_: ...

metrics:
  - metric:
      _target_: espnet3.systems.asr.metrics.wer.WER
    inputs:
      ref: ref
      hyp: hyp
```

### Metric interface

Metric classes are defined as `AbsMetrics` subclasses and are instantiated from
`measure.yaml`. See the core documentation for details:

- [ESPnet3 Metrics](../core/components/metrics.md)
