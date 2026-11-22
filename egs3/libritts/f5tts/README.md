# LibriTTS F5-TTS recipe

F5-TTS trained on LibriTTS and evaluated on the LibriSpeech-PC test-clean cross-sentence set, the evaluation set of the F5-TTS paper (arXiv 2410.06885).
`conf/training.yaml` is the F5-TTS Small configuration (158M parameters, character tokens).

## Data Preparation

`create_dataset` downloads the training corpus and the evaluation set into `downloads/` and builds every manifest:

- LibriTTS (OpenSLR 60): `train-clean-100`, `train-clean-360` and `train-other-500` for training, `dev-clean` for validation.
- LibriSpeech `test-clean` (OpenSLR 12) and the F5-TTS cross-sentence pair list: 1127 same-speaker prompt/target pairs.
  The pair list is pinned to a commit of the F5-TTS repository in `dataset/config.yaml`.

```bash
python run.py --stages create_dataset --training_config conf/training.yaml
```

To reuse corpora you already have, set `builder.dataset_path` in `dataset/config.yaml` to the directory that holds the `LibriTTS` and `LibriSpeech` folders.
A subset counts as prepared only once it carries a `.complete` marker, which `create_dataset` writes after extracting that subset.
A corpus staged any other way has no markers and would be downloaded again; create them yourself to prevent that (`touch <dataset_path>/<corpus>/<subset>/.complete`).

## Commands

Run the following commands from `egs3/libritts/f5tts/`.
`path.sh` activates the tools environment and puts this checkout on `PYTHONPATH`.

```bash
. ./path.sh

# 1) Download the corpora and build every manifest (run once), filter by
#    duration, build the token list, collect feature shapes
python run.py --stages create_dataset remove_long_short create_token_list collect_stats \
    --training_config conf/training.yaml

# 2) Train
python run.py --stages train --training_config conf/training.yaml

# 3) Synthesize the LibriSpeech-PC pairs
python run.py --stages infer \
    --training_config conf/training.yaml --inference_config conf/inference.yaml

# 4) Score
python run.py --stages measure \
    --training_config conf/training.yaml --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml

# 5) Pack and upload the model, then the demo
python run.py --stages pack_model upload_model \
    --training_config conf/training.yaml --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml --publication_config conf/publication.yaml
python run.py --stages pack_demo upload_demo \
    --training_config conf/training.yaml --demo_config conf/demo.yaml
```

See `egs3/TEMPLATE/f5tts/README.md` for the stages, `measure`'s dependencies and loading a packed model.

## Results

### F5-TTS Small

- Training config: `conf/training.yaml`
- Inference config: `conf/inference.yaml`
- Metrics config: `conf/metrics.yaml`

`conf/training.yaml` follows the official `F5TTS_Small` configuration: a 158M-parameter DiT on character tokens, trained on 4 V100 GPUs with 8 gradient accumulation steps, 307,200 mel frames per optimizer update.
`conf/inference.yaml` synthesizes from the EMA weights of `last.ckpt` with 32 ODE solver steps, guidance strength 2.0, sway sampling coefficient -1.0 and seed 0.

Scored by the `measure` stage through VERSA on the 1127 LibriSpeech-PC test-clean cross-sentence pairs:

| dataset | WER | CER | SIM | UTMOS |
| --- | --- | --- | --- | --- |
| LibriSpeech-PC test-clean | 1.64 | 0.76 | 0.706 | 4.13 |

WER and CER are corpus-level rates from faster-whisper large-v3, and SIM uses the `espnet/voxcelebs12_ecapa_wavlm_joint` speaker model, so they are not on the scale of the F5-TTS paper; `egs3/TEMPLATE/f5tts/conf/metrics.yaml` describes the differences from the official scorer.

#### Pretrained model

- Model: [`espnet/libritts_f5tts_training`](https://huggingface.co/espnet/libritts_f5tts_training)
- Demo Space: [`NewGame/libritts_f5tts_training`](https://huggingface.co/spaces/NewGame/libritts_f5tts_training)
