# ESPnet3 ASR template

This template uses `espnet3.systems.esp2_asr.ASRSystem` to train and evaluate an ESPnet2 ASR task model. It provides the standard dataset creation, tokenizer training, statistics collection, training, inference, measurement, and publication stages. Recipe code supplies the corpus-specific dataset and configuration.

## Data

Provide speech and transcription fields. The standard inference contract expects mono audio at 16 kHz; audio contracts can resample input as configured.

```yaml
fields:
  speech: audio
  text: text
```

The training configuration must identify an ESPnet2 ASR task/model and its tokenizer settings. The recipe's dataset builder determines how source data is prepared and cached.

## Metrics

The system provides WER, CER, and TER metrics. Configure references from the dataset and hypotheses from inference output. TER is recipe-specific: point it at the trained SentencePiece model in the metrics configuration.

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

## Typical workflow

Run these commands from the corpus recipe directory after setting its dataset and model configuration. The recipe's `run.py` uses `launch()` to apply the shared template defaults and recipe overrides.

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

For a runnable small example, see [mini_an4](../../mini_an4/esp2_asr). For a full recipe, see [LibriSpeech 100h](../../librispeech_100/esp2_asr).
