# ESPnet3 ASR template

## Overview

This template uses `espnet3.systems.esp2_asr.system.ASRSystem` for an ESPnet2 ASR task. Corpus recipes supply the dataset implementation and configuration overrides; this directory provides shared defaults.

## Data

A recipe's dataset class (`dataset/dataset.py`, a `BaseDataset` subclass) declares the fields each item carries as a class attribute. Both the LibriSpeech 100h and mini_an4 recipes use:

```python
from espnet3.contract import Field

fields = (Field("speech", "audio"), Field("text", "text"))
```

`fields` is declared on the dataset class, not in a configuration file. The default inference contract in `conf/inference.yaml` resamples input audio to 16 kHz (`rate_policy: resample`) and requires mono input: multichannel audio is rejected unless a normalizer such as `mean_channels` or `first_channel` is enabled.

The training configuration selects the ESPnet2 ASR task/model and tokenizer settings. Dataset download, preparation, and caching are recipe-specific.

## System Features

`ASRSystem` provides dataset creation, tokenizer training, statistics collection, training, inference, measurement, and model/demo packaging stages.

## Metrics

The system provides WER, CER, and TER. Bind references to the dataset transcript and hypotheses to inference output. TER scores the model's subword tokens, so it is commented out in the template's `conf/metrics.yaml`; enable it by setting `bpemodel` to the recipe's trained SentencePiece model.

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

Run from a corpus recipe directory after configuring its dataset and model. Replace `<training_config>` with the recipe-specific path. The recipe's `run.py` uses `launch()` to combine these defaults with recipe-specific settings.

```bash
python run.py --stages create_dataset train_tokenizer collect_stats \
  --training_config <training_config>
python run.py --stages train \
  --training_config <training_config>
python run.py --stages infer \
  --training_config <training_config> \
  --inference_config conf/inference.yaml
python run.py --stages measure \
  --training_config <training_config> \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml
```

For [mini_an4](../../mini_an4/esp2_asr/README.md), use `conf/training_asr_transformer.yaml`; for [LibriSpeech 100h](../../librispeech_100/esp2_asr/README.md), use `conf/tuning/training_e_branchformer.yaml`.
