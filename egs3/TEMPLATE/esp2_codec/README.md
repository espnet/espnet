# ESPnet3 neural codec recipe

Shared default configs, model card and demo app for recipes that train an
ESPnet2 GAN codec (Encodec, SoundStream, DAC, FunCodec) through
`espnet3.systems.esp2_codec.system.ESP2CodecSystem`.
A recipe adds its `dataset/` package (a `DatasetBuilder` writing the manifests
and a `Dataset` returning `{"speech": waveform}` items, plus `utt_id` and
`wav_path` in inference mode), its `conf/*.yaml` deltas over this template's
defaults (the codec, optimizers and dataloader in a training config, the
`dataset.test` entries in `inference.yaml` and `metrics.yaml`), a copy of
`src/app.py`, and a thin `run.py`; see `egs3/libritts/esp2_codec`.
The template's training config loads the recipe's `train` and `valid` splits
and runs `CodecPreprocessor`, which hands the `speech` field to the ESPnet2
codec model as `audio`.

## Quick start

```bash
# 1) Build the manifests (run once)
python run.py --stages create_dataset --training_config conf/training_encodec.yaml

# 2) Collect feature statistics
python run.py --stages collect_stats --training_config conf/training_encodec.yaml

# 3) Train; the ESPnet2 config is saved as exp/<tag>/config.yaml beside the
#    checkpoints, and inference rebuilds the codec from it
python run.py --stages train --training_config conf/training_encodec.yaml

# 4) Resynthesize the test sets (one WAV per utterance, listed in wav.scp)
python run.py --stages infer --training_config conf/training_encodec.yaml \
    --inference_config conf/inference.yaml

# 5) Score against the test set's own waveforms with VERSA
python run.py --stages measure --training_config conf/training_encodec.yaml \
    --inference_config conf/inference.yaml --metrics_config conf/metrics.yaml

# 6) Pack and upload the model
python run.py --stages pack_model upload_model \
    --training_config conf/training_encodec.yaml --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml --publication_config conf/publication.yaml

# 7) Pack and upload the Gradio demo
python run.py --stages pack_demo upload_demo \
    --training_config conf/training_encodec.yaml --demo_config conf/demo.yaml
```

`measure` needs `versa` (`tools/installers/install_versa.sh`). The signal
metrics also need `mir_eval`, and the pseudo-MOS predictors `speechmos` and
`onnxruntime`. The template's `metrics.yaml` declares the VERSA metric
(`espnet3.components.metrics.versa.VersaMetric`) with the scores it promises
under `outputs`, and the stage fails when VERSA drops one instead of
reporting a partial table.

## Using a packed model

```python
from espnet3.publication import InferenceModel

model = InferenceModel.from_pretrained("<hf_repo>")
audio = model("speech.wav")["wav"]  # an Audio: audio.array (float32) at audio.rate
```

The input may be a path, a `(rate, samples)` pair or an array; it is resampled
to the codec's rate and averaged to one channel before encoding, as declared
in `conf/inference.yaml`'s `contract`.
