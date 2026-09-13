# MELD speech emotion recognition recipe

Classifies each utterance into one of seven emotions (`neutral`, `joy`,
`surprise`, `anger`, `sadness`, `disgust`, `fear`) with a frozen
[WavLM Base+](https://github.com/microsoft/unilm/tree/master/wavlm) frontend,
a Transformer encoder and a linear head.

[MELD](https://affective-meld.github.io/) ships audio as MP4, so `ffmpeg` must be
on `PATH`. `create_dataset` downloads the corpus when it is not already present,
extracts it, converts every clip to 16 kHz mono WAV, and writes one manifest per
split. The conversion dominates the stage and takes over an hour.

Two environment variables control where the data lives. Both are optional and
default to `download/` and `data/` under the recipe.

- `MELD` — an existing MELD source tree, or where to download it to
- `MELD_OUTPUT` — where the converted WAV files and manifests are written

## Quick start

```bash
# 1) Fetch MELD, convert the audio, and write the manifests
python run.py --stages create_dataset remove_long_short prepare_labels \
    --training_config conf/training.yaml

# 2) Collect feature statistics
python run.py --stages collect_stats \
    --training_config conf/training.yaml

# 3) Train
python run.py --stages train \
    --training_config conf/training.yaml

# 4) Infer
python run.py --stages infer \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml

# 5) Score
python run.py --stages measure \
    --training_config conf/training.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`remove_long_short` filters the train and valid splits to 0.1–20 s; the test
split is left untouched so the reported scores cover the whole set.
`prepare_labels` writes `token_list` from the filtered training manifest.

## Results

`measure` writes all five metrics to `${inference_dir}/metrics.json`.

| | WA | UA | Macro F1 | mAP | AUC |
|---|---|---|---|---|---|
| test | 52.72 | 25.44 | 26.17 | 28.30 | 70.05 |
| valid | 48.55 | 26.07 | 26.07 | 30.45 | 69.57 |

MELD is heavily imbalanced — `neutral` is 47% of the training split — so WA alone
overstates how well a model separates the emotions. UA and Macro F1 weight every
class equally and are the more informative pair here.

Utterance-level alignment in MELD is known to be imperfect
([declare-lab/MELD#30](https://github.com/declare-lab/MELD/issues/30)), which
limits the accuracy any model can reach on it.

## Pretrained Models

- [`conf/training.yaml`](https://huggingface.co/espnet/meld_cls_wavlm_base_plus)
