# ESPnet3 audio classification template

## Overview

This template uses `espnet3.systems.esp2_cls.system.ESP2CLSSystem` for an ESPnet2 classification task. Corpus recipes provide the dataset implementation and task-specific configuration.

## Data

Provide audio and a class label for each example. The label vocabulary and ordering must match the model's class indices.

```yaml
fields:
  speech: audio
  label: text
```

The default inference contract resamples audio to 16 kHz, averages multichannel input to mono, and emits a predicted label (`hyp`) and class scores (`score`).

## System Features

The system prepares labels, filters examples by configured duration, and provides the common statistics, training, inference, measurement, and packaging stages.

## Metrics

The system provides WA, UA, macro F1, MAP, and AUC. Metrics read references from `dataset:label`; MAP and AUC also consume `score`. Set `token_list` to the model's class order for score-based metrics.

```yaml
metrics:
  - metric:
      _target_: espnet3.systems.esp2_cls.metrics.wa.WA
    inputs:
      ref: dataset:label
      hyp: hyp
```

## Commands

Run from the corpus recipe directory after configuring the dataset and classification model. `run.py` calls `launch()` to combine this template with recipe-specific overrides.

```bash
python run.py --stages create_dataset remove_long_short prepare_labels collect_stats \
  --training_config conf/training.yaml
python run.py --stages train \
  --training_config conf/training.yaml
python run.py --stages infer \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml
python run.py --stages measure \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

See [mini_an4](../../mini_an4/esp2_cls/README.md) for a small runnable example and [MELD](../../meld/esp2_cls/README.md) for a corpus recipe.
