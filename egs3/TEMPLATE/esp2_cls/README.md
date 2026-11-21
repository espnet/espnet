# ESPnet3 audio classification template

This template uses `espnet3.systems.esp2_cls.ESP2CLSSystem` for an ESPnet2 classification task. In addition to the common training and inference stages, the system prepares labels and supports removing examples outside configured duration limits. The recipe supplies its dataset builder and task-specific configuration.

## Data

Provide audio and a class label for each example. The label vocabulary and ordering must match the model's class indices.

```yaml
fields:
  speech: audio
  label: text
```

The default inference contract resamples audio to 16 kHz, averages multichannel input to mono, and emits a predicted label (`hyp`) and class scores (`score`).

## Metrics

The system provides weighted accuracy (WA), unweighted accuracy (UA), macro F1, mean average precision (MAP), and AUC metrics. Metrics read the reference from `dataset:label`; MAP and AUC also consume `score`. Set `token_list` to the class order used by the model when configuring score-based metrics.

```yaml
metrics:
  - metric:
      _target_: espnet3.systems.esp2_cls.metrics.wa.WA
    inputs:
      ref: dataset:label
      hyp: hyp
```

Use the metric class names and required options exposed by the recipe's installed ESPnet version when enabling the other supported classification metrics.

## Typical workflow

Run from the corpus recipe directory after configuring the dataset and classification model. `run.py` calls `launch()` to combine this template with recipe-specific overrides.

```bash
python run.py --stages create_dataset remove_long_short prepare_labels collect_stats train \
  --training_config conf/training.yaml
python run.py --stages infer \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml
python run.py --stages measure \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

See [mini_an4](../../mini_an4/esp2_cls) for a small runnable example and [MELD](../../meld/esp2_cls) for a corpus recipe.
