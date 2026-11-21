# ESPnet3 F5-TTS template

This template uses the native `espnet3.systems.f5tts.F5TTSSystem`, rather than an ESPnet2 task. The system provides duration filtering, token-list preparation, statistics collection, training, inference, and model/demo packaging. It does not currently provide built-in metrics; define recipe-specific metrics if evaluation is needed.

## Data

Training examples need speech and text. The dataset builder and dataset implementation are recipe-specific, as are the cache layout and text normalization choices.

```yaml
fields:
  speech: audio
  text: text
```

Inference conditions on target text and reference speech, and produces generated audio (`wav`). Set the recipe's inference configuration to provide the inputs required by its selected F5-TTS model.

## Configuration

The template intentionally leaves dataset, model, optimizer, and dataloader choices to the recipe. Configure them in the corpus recipe's `conf/training.yaml`; use its inference configuration for generation settings.

## Typical workflow

Run from the corpus recipe directory after filling in those settings. Its `run.py` calls `launch()` with `F5TTSSystem` and the recipe-specific configuration.

```bash
python run.py --stages create_dataset remove_long_short create_token_list collect_stats train \
  --training_config conf/training.yaml
python run.py --stages infer \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml
```
