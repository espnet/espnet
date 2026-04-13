---
title: ESPnet3 Config Overview
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Config Overview

ESPnet3 uses separate YAML files per stage. Most recipes follow the layout:

```
egs3/<recipe>/<task>/conf/
  train.yaml
  infer.yaml
  measure.yaml
  publish.yaml
  demo.yaml
```

Each file is passed to `run.py` via the matching CLI flag:

```bash
python run.py \
  --train_config conf/train.yaml \
  --infer_config conf/infer.yaml \
  --measure_config conf/measure.yaml \
  --publish_config conf/publish.yaml \
  --demo_config conf/demo.yaml
```

## Resolvers

ESPnet3 registers custom OmegaConf resolvers for loading external files from
YAML (for example, pulling vocab or values from another config). See
[Resolvers](./resolvers.md) for details.

## Stage to config mapping

| Stage | Config flag | Typical file |
| --- | --- | --- |
| create_dataset | `--train_config` | `train.yaml` |
| collect_stats | `--train_config` | `train.yaml` |
| train | `--train_config` | `train.yaml` |
| infer | `--infer_config` | `infer.yaml` |
| measure | `--measure_config` | `measure.yaml` |
| pack_model | `--train_config` | `train.yaml` |
| upload_model | `--publish_config` | `publish.yaml` |
| pack_demo | `--demo_config` | `demo.yaml` |
| upload_demo | `--demo_config` | `demo.yaml` |

## What goes in each config

| File | Purpose | Typical contents |
| --- | --- | --- |
| [`train.yaml`](./train_config.md) | Training pipeline | model, trainer, optimizers, dataloader, exp_dir |
| [`infer.yaml`](./infer_config.md) | Inference/decoding | model entrypoint, dataset, infer_dir, output_fn, parallel |
| [`measure.yaml`](./measure_config.md) | Scoring/metrics | metrics, infer_dir, test sets |
| [`publish.yaml`](./publish_config.md) | Packaging/upload | pack settings, artifacts to include, HF upload options |
| [`demo.yaml`](./demo_config.md) | Demo build | UI spec, output mapping, infer config path, assets |

### Notes

- Train stage: [Train configuration](./train_config.md)
- Inference config: [Inference configuration](./infer_config.md)
- Metrics pipeline: [Measurement](../stages/measure.md)
- Demo customization: [Demo guide](../stages/demo.md)
- Resolvers: [Resolvers](./resolvers.md)
