# ESPnet3 ASR template

## Overview

This template uses `espnet3.systems.esp2_asr.ASRSystem` for an ESPnet2 ASR task. Corpus recipes supply the dataset implementation and configuration overrides; this directory provides shared defaults.

## Data

Provide speech and transcription fields. The default inference contract resamples audio to 16 kHz and requires mono input.

```yaml
fields:
  speech: audio
  text: text
```

The training configuration selects the ESPnet2 ASR task/model and tokenizer settings. Dataset download, preparation, and caching are recipe-specific.

## System Features

`ASRSystem` provides dataset creation, tokenizer training, statistics collection, training, inference, measurement, and model/demo packaging stages.

## Metrics

The system provides WER, CER, and TER. Bind references to the dataset transcript and hypotheses to inference output. TER is recipe-specific and requires the trained SentencePiece model path.

```yaml
metrics:
  - metric:
      _target_: espnet3.systems.esp2_asr.metrics.wer.WER
    inputs:
      ref: dataset:text
      hyp: text
  - metric:
      _target_: espnet3.systems.esp2_asr.metrics.cer.CER
    inputs:
      ref: dataset:text
      hyp: text
```

## Commands

Run from a corpus recipe directory after configuring its dataset and model. The recipe's `run.py` uses `launch()` to combine these defaults with recipe-specific settings.

```bash
python run.py --stages create_dataset train_tokenizer collect_stats train \
  --training_config conf/training.yaml
python run.py --stages infer \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml
python run.py --stages measure \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

For a small runnable example, see [mini_an4](../../mini_an4/esp2_asr/README.md). For a full recipe, see [LibriSpeech 100h](../../librispeech_100/esp2_asr/README.md).
