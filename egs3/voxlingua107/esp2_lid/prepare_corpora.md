# Prepare additional LID datasets

Run these commands from the ESPnet repository root. Each script writes
16 kHz audio and `<output-dir>/<split>/manifest.tsv`.

FLEURS and ML-SUPERB require `huggingface_hub`, `hf_xet`, and `pyarrow`;
VoxPopuli requires `fsspec` and `aiohttp`.

```bash
export HF_XET_HIGH_PERFORMANCE=1

# FLEURS
python -m egs3.voxlingua107.esp2_lid.src.fleurs_prepare \
  --source-dir /corpora/fleurs --output-dir egs3/voxlingua107/esp2_lid/data/fleurs \
  --languages ja_jp en_us --splits test

# ML-SUPERB 2.0
python -m egs3.voxlingua107.esp2_lid.src.ml_superb2_prepare \
  --source-dir /corpora/ml_superb2 --output-dir egs3/voxlingua107/esp2_lid/data/ml_superb2 \
  --splits dev dev_dialect

# VoxPopuli
python -m egs3.voxlingua107.esp2_lid.src.voxpopuli_prepare \
  --source-dir /corpora/voxpopuli --output-dir egs3/voxlingua107/esp2_lid/data/voxpopuli \
  --languages en --splits test

# Babel: supply locally obtained LDC data
python -m egs3.voxlingua107.esp2_lid.src.babel_prepare \
  --source-dir /corpora/babel/conversational/dev \
  --language asm --split dev --output-dir egs3/voxlingua107/esp2_lid/data/babel_asm
```

Babel's source directory must contain `audio/` and `transcription/`.
SPHERE audio requires `sph2pipe` on PATH. Other datasets download automatically.
Use `--help` for supported languages, splits, and other options.

## Evaluation and training

From the LID recipe directory, set `prepared_name` and `prepared_manifest`
in `conf/inference_prepared.yaml`, then evaluate:

```bash
python run.py --stages infer measure \
  --training_config conf/training.yaml \
  --inference_config conf/inference_prepared.yaml \
  --metrics_config conf/metrics.yaml
```

Evaluation selects languages in the model's training language inventory.
For combined training, add entries to `dataset.train` using the same `data_src`
and training manifests, omit the `lang2utt` filter, and rerun `collect_stats`.
