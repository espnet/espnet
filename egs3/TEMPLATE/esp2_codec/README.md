# ESPnet3 neural codec template

## Overview

This template uses `espnet3.systems.esp2_codec.system.ESP2CodecSystem` to train an ESPnet2 GAN codec (Encodec, SoundStream, DAC, FunCodec).
It provides the shared default configs, the model card and the demo app.
A corpus recipe adds its `dataset/` package, its `conf/*.yaml` deltas over these defaults (the codec, optimizers and dataloader in a training config, the `dataset.test` entries in `inference.yaml` and `metrics.yaml`), a copy of `src/app.py`, and a thin `run.py`; see `egs3/libritts/esp2_codec`.

## Data

A recipe's `dataset/` package holds a `DatasetBuilder` writing the manifests and a `Dataset` returning `{"speech": waveform}` items, plus `utt_id` and `wav_path` in inference mode.
The template's training config loads the recipe's `train` and `valid` splits and runs `CodecPreprocessor`, which hands the `speech` field to the ESPnet2 codec model as `audio`.

Inference takes anything that carries audio: a path, a `(rate, samples)` pair or an array.
It is resampled to the codec's rate and averaged to one channel before encoding, as declared in `conf/inference.yaml`'s `contract`, and the resynthesis is returned as `wav`.

## System Features

`ESP2CodecSystem` provides dataset creation, statistics collection, GAN training, inference, measurement, and model/demo packaging stages.
GAN training alternates a generator and a discriminator turn per batch with one named optimizer each (`optimizers` / `schedulers` in the training config); `trainer.gan` sets the turn order and the discriminator skip probability.
Inference and publication both read `exp/<tag>/last.ckpt`, the link the trainer keeps to the final checkpoint of a run.

## Metrics

The template's `conf/metrics.yaml` declares the VERSA metric (`espnet3.components.metrics.versa.VersaMetric`), which scores each resynthesized utterance against the test set's own waveform.
It lists the scores it promises under `outputs`, and the stage fails when VERSA drops one instead of reporting a partial table.
`measure` needs `versa` (`tools/installers/install_versa.sh`).
The signal metrics also need `mir_eval`, and the pseudo-MOS predictors `speechmos` and `onnxruntime`.

## Commands

Run from a corpus recipe directory.
Replace `<training_config>` with the recipe-specific path, for example `conf/training_encodec.yaml`.

```bash
# 1) Build the manifests (run once)
python run.py --stages create_dataset --training_config <training_config>

# 2) Collect feature statistics
python run.py --stages collect_stats --training_config <training_config>

# 3) Train; the ESPnet2 config is saved as exp/<tag>/config.yaml beside the
#    checkpoints, and inference rebuilds the codec from it
python run.py --stages train --training_config <training_config>

# 4) Resynthesize the test sets (one WAV per utterance, listed in wav.scp)
python run.py --stages infer --training_config <training_config> \
    --inference_config conf/inference.yaml

# 5) Score against the test set's own waveforms with VERSA
python run.py --stages measure --training_config <training_config> \
    --inference_config conf/inference.yaml --metrics_config conf/metrics.yaml

# 6) Pack and upload the model
python run.py --stages pack_model upload_model \
    --training_config <training_config> --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml --publication_config conf/publication.yaml

# 7) Pack and upload the Gradio demo
python run.py --stages pack_demo upload_demo \
    --training_config <training_config> --demo_config conf/demo.yaml
```

A packed model loads without the recipe:

```python
from espnet3.publication import InferenceModel

model = InferenceModel.from_pretrained("<hf_repo>")
audio = model("speech.wav")["wav"]  # an Audio: audio.array (float32) at audio.rate
```
