# ESPnet3 audio classification template

## Overview

This template uses `espnet3.systems.esp2_cls.system.ESP2CLSSystem` for an ESPnet2 classification task. Corpus recipes provide the dataset implementation and task-specific configuration.

## Data

Each item carries audio and a class label. A recipe's dataset class (`dataset/dataset.py`, a `BaseDataset` subclass) declares them as a class attribute; both the MELD and mini_an4 recipes use:

```python
from espnet3.contract import Field

fields = (Field("speech", "audio"), Field("label", "text"))
```

`fields` is declared on the dataset class, not in a configuration file. The `prepare_labels` stage writes the class list to `${data_dir}/token_list`, which fixes the class order the model and the score-based metrics use.

The default inference contract resamples audio to 16 kHz, averages multichannel input to mono (`mean_channels`), and emits a predicted label (`hyp`) and class scores (`score`).

## System Features

The system prepares labels, filters examples by configured duration, and provides the common statistics, training, inference, measurement, and packaging stages.

## Metrics

The system provides WA, UA, macro F1, MAP, and AUC. Metrics read references from `dataset:label`; MAP and AUC also consume `score`. MAP and AUC require `token_list` (the class order written by `prepare_labels`); UA and macro F1 accept it optionally to fix the class set.

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
