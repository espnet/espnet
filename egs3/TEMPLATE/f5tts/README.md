# ESPnet3 F5-TTS template

## Overview

This template uses the native `espnet3.systems.f5tts.system.F5TTSSystem`, rather than an ESPnet2 task. Corpus recipes provide the dataset implementation and model/training choices.

## Data

Training examples need speech and text. Dataset preparation, cache layout, and text normalization are recipe-specific. Inference conditions on target text and reference speech, and produces generated audio (`wav`). When reference speech contains different text, provide its transcript as `reference_text`; it is used for conditioning and duration estimation. If omitted, `F5TTSInference` uses the target text, which is appropriate when it matches the reference speech. Configure the recipe's inference inputs for its selected model.

```yaml
fields:
  speech: audio
  text: text
```

## System Features

The system provides dataset creation, duration filtering, token-list preparation, statistics collection, training, inference, and model/demo packaging.

## Metrics

No built-in F5-TTS metrics are currently provided. Add recipe-specific metrics when evaluation is needed.

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
