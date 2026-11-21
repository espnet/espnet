# ESPnet3 espnet2 GAN-TTS recipe template

Starter files for a recipe that trains an espnet2 GAN-based TTS model (VITS,
JETS, ...) through `espnet3.systems.esp2_gan_tts.system.ESP2GANTTSSystem`.
Copy `egs3/libritts/esp2_gan_tts/` when starting a new corpus; the configs
under `conf/` here are the defaults every recipe config is merged over, and
`src/app.py` is the Gradio launcher `pack_demo` ships.

## Quick start

```bash
# 0) Edit configs to set paths.
#    Keep `conf/training.yaml:data_dir` as the canonical dataset location.
#    When `--training_config` is also passed to `infer` or `measure`, run.py
#    propagates experiment path fields from training into inference/metrics.
#    Standalone inference or metrics configs must define their own `exp_tag`
#    or `exp_dir`.

# 1) Build manifests (run once)
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Extract speaker embeddings, filter by duration, build the token list
python run.py --stages compute_xvectors  --training_config conf/training.yaml
python run.py --stages remove_long_short --training_config conf/training.yaml
python run.py --stages create_token_list --training_config conf/training.yaml

# 3) Collect feature stats
python run.py --stages collect_stats --training_config conf/training.yaml

# 4) Train
python run.py --stages train --training_config conf/training.yaml

# 5) Synthesize
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 6) Score
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml

# 7) Pack and upload the model
python run.py --stages pack_model \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --publication_config conf/publication.yaml
python run.py --stages upload_model \
    --training_config conf/training.yaml \
    --publication_config conf/publication.yaml

# 8) Pack and upload the Gradio demo
python run.py --stages pack_demo   --training_config conf/training.yaml --demo_config conf/demo.yaml
python run.py --stages upload_demo --training_config conf/training.yaml --demo_config conf/demo.yaml
```

The stage list, its order and the config role each stage runs on are
declared once, as `ESP2GANTTSSystem.stages`; a recipe's `run.py` is
`launch(ESP2GANTTSSystem)`. Stages always execute in that order, whatever
order `--stages` lists them in, so `--stages all` runs the whole pipeline.

## Inference contract

`conf/inference.yaml` declares what the model takes and returns, checked
around every call by the `infer` stage, by `InferenceModel` on a packed bundle
and by the demo: the raw transcript (`text`) and the speaker embedding
(`spembs`, a 1-D vector) in, the synthesized waveform (`wav`, mono at 22.05
kHz by default) out. `Text2Speech` tokenizes the raw text with the cleaner,
g2p and token list recorded in the espnet2-style config `train` writes to
`${exp_dir}/config.yaml`, so no preprocessor runs at inference. `measure`
reads its references (the ground-truth recording and transcript) from the
test set itself.

## Using a packed model

```python
from espnet3.publication import InferenceModel

model = InferenceModel.from_pretrained("<hf_repo>")
audio = model({"text": "The text to synthesize.", "spembs": spembs})["wav"]
# an Audio: audio.array (float32) at audio.rate
```

`spembs` is the speaker embedding the model was trained on, a 1-D vector from the extractor named under `xvector` in `conf/training.yaml`.
`build_xvector_extractor` in `src/app.py` shows how the demo computes it from a reference recording.
The contract in `conf/inference.yaml` checks both inputs and the output, so a mismatched embedding shape or an empty text is refused before the model runs.

## Demo

`src/app.py` shows a text box and a reference-recording widget (`ui.inputs` in
`conf/demo.yaml`), turns the recording into the speaker embedding with the
extractor named under `xvector` in `conf/demo.yaml` (keep it equal to the
training config's `xvector` block) and calls the packed model through the
inference contract. A recipe overrides `xvector` and adds the extractor's
package to `pack.requirements`; see `egs3/libritts/esp2_gan_tts/conf/demo.yaml`.
