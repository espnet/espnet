# LibriTTS VITS recipe

Multi-speaker English TTS on LibriTTS using VITS with x-vector speaker conditioning, run through `espnet3.systems.esp2_gan_tts.system.ESP2GANTTSSystem`.
The configs under `conf/` are merged over `egs3/TEMPLATE/esp2_gan_tts/conf/`, so they only list what this recipe changes.

## Data Preparation

`create_dataset` downloads the OpenSLR subsets listed in `dataset/config.yaml` into `downloads/LibriTTS/`, resamples the audio to 22.05 kHz under `data/` and builds the per-split TSV manifests:

```bash
python run.py --stages create_dataset --training_config conf/training.yaml
```

Each finished extraction is marked with `downloads/LibriTTS/<subset>/.complete`.
If you already have LibriTTS on disk, symlink or copy it to `downloads/LibriTTS/` and create that marker for every subset yourself (for example `touch downloads/LibriTTS/train-clean-100/.complete`); without it the stage treats the subset as unfinished and downloads it again.

`compute_xvectors` then extracts one speaker embedding per utterance.
This model was trained against SpeechBrain ECAPA-TDNN embeddings, so `conf/training.yaml` pins `xvector.toolkit: speechbrain` (needs `pip install speechbrain`).
The template default is espnet's own `espnet/voxcelebs12_rawnet3` extractor, which needs no extra dependency; switching requires retraining with a matching `spk_embed_dim`.

## Commands

Run the following commands from `egs3/libritts/esp2_gan_tts/`.
`path.sh` activates the tools environment and puts this checkout on `PYTHONPATH`.

```bash
. ./path.sh

# 1) Download LibriTTS and build per-split TSV manifests (run once)
python run.py --stages create_dataset --training_config conf/training.yaml

# 2) Extract x-vector speaker embeddings (one .pt file per utterance)
python run.py --stages compute_xvectors --training_config conf/training.yaml

# 3) Filter utterances by duration
python run.py --stages remove_long_short --training_config conf/training.yaml

# 4) Build the phoneme token list
python run.py --stages create_token_list --training_config conf/training.yaml

# 5) Collect feature statistics (resumable: set collect_stats.num_shards>1)
python run.py --stages collect_stats --training_config conf/training.yaml

# 6) Train VITS
python run.py --stages train --training_config conf/training.yaml

# 7) Synthesize from test text
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 8) Compute the metrics
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml

# 9) Bundle the trained model for release (packs the phoneme token list too)
python run.py --stages pack_model \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --publication_config conf/publication.yaml

# 10) Upload it to the Hugging Face Hub (run `hf auth login` first)
python run.py --stages upload_model \
    --training_config conf/training.yaml \
    --publication_config conf/publication.yaml

# 11) Pack the Gradio demo, then upload it as a Space
python run.py --stages pack_demo   --training_config conf/training.yaml --demo_config conf/demo.yaml
python run.py --stages upload_demo --training_config conf/training.yaml --demo_config conf/demo.yaml
```

Stages always execute in this order, whatever order `--stages` lists them in, so `--stages all` runs the whole pipeline.

`infer` passes each test transcript and its x-vector to `Text2Speech`, which tokenizes the raw text with the phoneme token list, cleaner and g2p recorded in `exp/${exp_tag}/config.yaml`, and writes one 22.05 kHz WAV per utterance under `exp/${exp_tag}/inference/<test>/` listed in `wav.scp`.
`measure` runs [versa](https://github.com/wavlab-speech/versa), which has to be installed in the same environment, and reads the ground-truth recording and transcript of each utterance from the test set itself (`dataset:wav_path` and `dataset:raw_text` in the template's `conf/metrics.yaml`).

`pack_demo` ships the template's `src/app.py`.
Because this recipe trains a multi-speaker VITS, the demo takes **text plus a reference recording**: it runs the same SpeechBrain ECAPA extractor as the `compute_xvectors` stage to turn that recording into the `spembs` the model needs, then synthesizes the text in that voice through the packed model's inference contract.
`conf/demo.yaml` names that extractor under `xvector` and adds `speechbrain` to the Space's requirements; if you retrain against a different embedding model, change both to match, or the embedding the demo builds will not match the space the model was trained in.

## Results

### VITS

- Training config: `conf/training.yaml`
- Inference config: `conf/inference.yaml`
- Metrics config: `conf/metrics.yaml`

The published model is the `step577710` checkpoint of this recipe's VITS configuration, scored by the `measure` stage through VERSA on the LibriTTS test set (4,507 utterances):

| dataset | whisper_wer | whisper_cer | spk_similarity | utmos | dns_overall | dns_p808 |
| --- | --- | --- | --- | --- | --- | --- |
| test | 12.70 | 7.12 | 0.505 | 2.98 | 3.19 | 3.79 |

That checkpoint was trained on LibriTTS's native 24 kHz audio with the 22.05 kHz model configuration, so its bundle declares a 24 kHz output and these scores are of 24 kHz audio.
The recipe as checked in resamples the corpus to 22.05 kHz in `create_dataset`; the results above do not reproduce that configuration, and a retrain at that rate replaces the published model.

#### Pretrained model

- Model: [`NewGame/libritts_esp2_gan_tts_train_vits_libritts`](https://huggingface.co/NewGame/libritts_esp2_gan_tts_train_vits_libritts)

The model loads directly once you have a speaker embedding for the target voice.
`spembs` is a SpeechBrain `spkrec-ecapa-voxceleb` embedding (192 values) of a reference recording, the same extractor `compute_xvectors` runs; `build_xvector_extractor` in `egs3/TEMPLATE/esp2_gan_tts/src/app.py` computes it from a `(rate, samples)` pair.

```python
from espnet3.publication import InferenceModel

model = InferenceModel.from_pretrained("NewGame/libritts_esp2_gan_tts_train_vits_libritts")
audio = model({"text": "He hoped there would be stew for dinner.", "spembs": spembs})["wav"]
# an Audio: audio.array (float32) at audio.rate (24000)
```

The demo built by `pack_demo` from `conf/demo.yaml` takes text plus a reference recording and computes that embedding itself.
