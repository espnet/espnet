# ESPnet3 F5-TTS template

## Overview

This template uses the native `espnet3.systems.f5tts.system.F5TTSSystem`, rather than an ESPnet2 task, to train `espnet3.systems.f5tts.src.f5tts.F5TTS`.
It provides the shared default configs, the model card and the demo app.
A corpus recipe adds its `dataset/` package, its `conf/*.yaml` deltas over these defaults (the model, data and number of training steps in `training.yaml`, the `dataset.test` entries and sampling settings in `inference.yaml`), a copy of `src/app.py`, and a thin `run.py`; see `egs3/libritts/f5tts`.

## Data

Training items carry speech and its transcript, the two inputs `F5TTS.forward` consumes.
A recipe's `DatasetBuilder` writes tab-separated manifests whose first three columns are `utt_id`, `wav_path` and `text`, and its dataset class (`dataset/dataset.py`, a `BaseDataset` subclass) declares the fields it serves:

```python
from espnet3.contract import Field

fields = (Field("speech", "audio"), Field("text", "text"))
```

Dataset preparation, cache layout, and text normalization are recipe-specific.

Inference conditions on the target `text` and a reference recording, and returns the generated audio as `wav`.
`F5TTSInference` takes the reference as `reference_speech` with its transcript `reference_text`, and all three inputs are required: the model reads the reference through its transcript and sets the output duration from the reference's speech rate.
The template's `conf/inference.yaml` names the model, sets `input_key: [text, reference_speech, reference_text]` and declares the contract: the reference may be a path, a `(rate, samples)` pair or an array, and it is resampled to 24 kHz and reduced to its first channel before the model sees it.
An evaluation dataset therefore serves `text`, `reference_speech`, `reference_text`, `utt_id` and `ref_wav_path`, the prompt path the speaker similarity is scored against.

## System Features

The system provides dataset creation, duration filtering (`remove_long_short`), token-list preparation (`create_token_list`), statistics collection, training, inference, measurement, and model/demo packaging.
The template's `conf/training.yaml` carries the F5-TTS optimizer defaults: AdamW at 7.5e-5 with a 20,000-step linear warmup followed by a linear decay over `trainer.max_steps`, which each recipe sets.
The model, the data and the dataloader are left to the recipe.
`train` saves the training config as `exp/<tag>/config.yaml` beside the checkpoints, and inference rebuilds the model from it.

## Metrics

The template's `conf/metrics.yaml` scores WER, speaker similarity and UTMOS through VERSA (`espnet3.components.metrics.versa.VersaMetric`), and says how each score relates to the official F5-TTS scorer.
`measure` needs `versa` (`tools/installers/install_versa.sh`), `faster-whisper` (VERSA's `tools/install_fwhisper.sh`), `openai-whisper` (the text cleaner) and `s3prl` (the speaker model's front-end).
The defaults score on a GPU (`use_gpu: true`, faster-whisper `compute_type: float16`); on a CPU-only machine override the metric in your recipe's `metrics.yaml` with `use_gpu: false` and a CPU `compute_type` (faster-whisper's `int8` or `float32`).

## Commands

Run from a corpus recipe directory after setting its dataset, model and `trainer.max_steps`; its `run.py` calls `launch()` with `F5TTSSystem`.

```bash
# 1) Build manifests (run once)
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Filter by duration, then build the token list from the filtered manifest
python run.py --stages remove_long_short create_token_list \
    --training_config conf/training.yaml

# 3) Collect feature shapes for batching
python run.py --stages collect_stats --training_config conf/training.yaml

# 4) Train
python run.py --stages train --training_config conf/training.yaml

# 5) Synthesize the test sets
python run.py --stages infer --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 6) Score (WER, speaker similarity, UTMOS through VERSA)
python run.py --stages measure --training_config conf/training.yaml \
    --inference_config conf/inference.yaml --metrics_config conf/metrics.yaml

# 7) Pack and upload the model
python run.py --stages pack_model upload_model \
    --training_config conf/training.yaml --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml --publication_config conf/publication.yaml

# 8) Pack and upload the Gradio demo
python run.py --stages pack_demo upload_demo \
    --training_config conf/training.yaml --demo_config conf/demo.yaml
```

A packed model loads without the recipe:

```python
from espnet3.publication import InferenceModel

model = InferenceModel.from_packed("exp/training/model_pack")  # or .from_pretrained(tag)
output = model(
    {
        "text": "The text to synthesize.",
        "reference_speech": "reference.wav",  # a path, or a (rate, samples) pair
        "reference_text": "The reference transcript.",
    }
)
audio = output["wav"]  # an Audio: audio.array (float32) at audio.rate
```
