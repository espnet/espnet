# ESPnet3 F5-TTS template

## Overview

This template uses the native `espnet3.systems.f5tts.system.F5TTSSystem`, rather than an ESPnet2 task. Corpus recipes provide the dataset implementation and model/training choices.

## Data

Training items carry speech and its transcript, the two inputs `F5TTS.forward` consumes. A recipe's dataset class (`dataset/dataset.py`, a `BaseDataset` subclass) declares them as a class attribute:

```python
from espnet3.contract import Field

fields = (Field("speech", "audio"), Field("text", "text"))
```

`fields` is declared on the dataset class, not in a configuration file. Dataset preparation, cache layout, and text normalization are recipe-specific.

Inference conditions on the target `text` and a reference recording, and returns the generated audio as `wav`. `F5TTSInference` takes the reference as `reference_speech` with its transcript `reference_text` (cross- or same-speaker protocol); when `reference_text` is omitted it defaults to the target text, which is only appropriate when the reference recording says the same thing. Passing `speech` alone uses it as a self-reference. The template's `conf/inference.yaml` leaves `model`, `input_key`, and the contract empty, so a recipe sets `input_key` to the keys its test set provides (for example `[text, reference_speech, reference_text]`) and declares its own contract.

## System Features

The system provides dataset creation, duration filtering, token-list preparation, statistics collection, training, inference, and model/demo packaging.

## Metrics

No built-in F5-TTS metrics are currently provided; the template's `conf/metrics.yaml` declares an empty `metrics` list. Add recipe-specific metrics when evaluation is needed.

## Commands

The template intentionally leaves dataset, model, optimizer, and dataloader choices to each recipe's `conf/training.yaml`. Run from that recipe directory after setting those values; its `run.py` calls `launch()` with `F5TTSSystem`.

```bash
python run.py --stages create_dataset remove_long_short create_token_list collect_stats \
  --training_config conf/training.yaml
python run.py --stages train \
  --training_config conf/training.yaml
python run.py --stages infer \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml
```
