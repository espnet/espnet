# ESPnet3 F5-TTS recipe

Shared runner, default configs and demo app for recipes that train
`espnet3.systems.f5tts.f5tts.F5TTS`.
A recipe adds its `dataset/` package (a `DatasetBuilder` writing tab-separated
manifests whose first three columns are `utt_id`, `wav_path` and `text`, and a
`Dataset` reading them), its `conf/*.yaml` deltas over this template's
defaults (the model, data and schedule in `training.yaml`, the `dataset.test`
entries and sampling settings in `inference.yaml`), a copy of `src/app.py`,
and a thin `run.py`; see `egs3/libritts/f5tts`.

## Quick start

```bash
# 1) Build manifests (run once)
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Filter by duration, then build the token list from the filtered manifest
python run.py --stages remove_long_short create_token_list \
    --training_config conf/training.yaml

# 3) Collect feature shapes for batching
python run.py --stages collect_stats --training_config conf/training.yaml

# 4) Train; the training config is saved as exp/<tag>/config.yaml beside the
#    checkpoints, and inference rebuilds the model from it
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

`measure` needs `versa` (`tools/installers/install_versa.sh`), `faster-whisper`
(VERSA's `tools/install_fwhisper.sh`), `openai-whisper` (the text cleaner) and
`s3prl` (the speaker model's front-end); `conf/metrics.yaml` says how the
scores relate to the official F5-TTS scorer. Its defaults score on a GPU
(`use_gpu: true`, faster-whisper `compute_type: float16`); on a CPU-only
machine override the metric in your recipe's `metrics.yaml` with
`use_gpu: false` and a CPU `compute_type` (faster-whisper's `int8` or
`float32`).

## Using a packed model

```python
from espnet3.api.inference import load

model = load("exp/training/model_pack")  # or a Hugging Face tag
output = model("The text to synthesize.", "reference.wav", "The reference transcript.")
samples, sample_rate = output["wav"].array, output["wav"].rate
```
