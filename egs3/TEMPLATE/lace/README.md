# ESPnet3 LACE template

## Overview

This template uses `espnet3.systems.lace.system.LACESystem` to fine-tune a published ESPnet2 GAN codec with the LACE (Layer-Adaptive Codec Encoding) quantizer wrapper, so every codebook layer runs at its own frame rate.
The model is described in "LACE: Layer-Wise Compression for Dynamic Frame Rate Codecs" (Trachu, Cornell, Chen, Watanabe, SLT 2026, https://arxiv.org/abs/2609.17509).
It provides the shared default configs, the model card and the demo app; see `egs3/libritts/lace` for a complete recipe.

## Data

A recipe's `dataset/` package builds the manifests and returns `{"audio": waveform}` items, matching `CommonPreprocessor(speech_name="audio")`.
`conf/training.yaml:model.pretrained_model_tag` names the published codec bundle to fine-tune (or `model.pretrained_model_dir` for a local `pack_model` output), and `model.compression_model` picks the segmentation.

Inference takes anything that carries audio: a path, a `(rate, samples)` pair or an array.
It is resampled to the codec's rate and averaged to one channel before encoding, as declared in `conf/inference.yaml`'s `contract`, and the resynthesis is returned as `resyn_audio`.

## System Features

`LACESystem` provides dataset creation, statistics collection, GAN training, inference, measurement, and model/demo packaging stages.
The training config has no `task:` key on purpose: the model is a Hydra target (`espnet3.systems.lace.src.lace.LACE`), and `train` writes its spec to `${exp_dir}/lace_model.yaml` for `infer` to rebuild.
`collect_stats` is not needed with the chunk-based loader, which uses no shape files.
When `--training_config` is also passed to `infer` or `measure`, `run.py` propagates the experiment path fields from training into inference and metrics.

## Metrics

The template's `conf/metrics.yaml` leaves `metrics` empty; a recipe declares the VERSA metric (`espnet3.components.metrics.versa.VersaMetric`), which scores each resynthesized utterance against the test set's own waveform.
A recipe's `metrics.yaml` lists the scores its configuration promises under `outputs`, and the stage fails when VERSA drops one instead of reporting a partial table.
`measure` needs `versa` (`tools/installers/install_versa.sh`).

## Commands

Run from a corpus recipe directory after editing its configs to set paths and pick the pretrained codec.

```bash
# 1) Build the manifests (run once)
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Fine-tune the pretrained codec with the LACE wrapper
python run.py --stages train --training_config conf/training.yaml

# 3) Resynthesize the test set at `model.rate` of conf/inference.yaml
#    (one WAV per utterance, listed in resyn_audio.scp)
python run.py --stages infer --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 4) Score the resynthesis against the test set's own waveforms with VERSA
python run.py --stages measure --training_config conf/training.yaml \
    --inference_config conf/inference.yaml --metrics_config conf/metrics.yaml

# 5) Pack and upload the model
python run.py --stages pack_model upload_model \
    --training_config conf/training.yaml --inference_config conf/inference.yaml \
    --publication_config conf/publication.yaml

# 6) Pack and upload the Gradio demo
python run.py --stages pack_demo upload_demo \
    --training_config conf/training.yaml --demo_config conf/demo.yaml
```

A packed model loads without the recipe:

```python
from espnet3.publication import InferenceModel

model = InferenceModel.from_pretrained("<hf_repo>")
audio = model("speech.wav")["resyn_audio"]  # an Audio: audio.array (float32) at audio.rate

# Any LACEAudioCoding argument replaces the packed one, which is how a rate sweep reads:
model = InferenceModel.from_pretrained("<hf_repo>", rate=0.25)
```
