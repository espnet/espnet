# Prepare additional LID corpora

These scripts start from downloaded corpus audio/annotations or Hugging Face
Parquet shards, following the corresponding ESPnet2 ASR data preparation.
They do not require ESPnet2-generated Kaldi data directories. VoxLingua107
already has its own `create_dataset` stage and needs no extra preparation script.

Run the commands from the ESPnet repository root. Outputs contain
`<split>/manifest.tsv` (utterance ID, absolute WAV path, ISO-639-3 language),
16 kHz mono WAVs, and `source.json` recording the source and selected split.
Source audio is preserved. Use different output directories for different
corpora or preparation settings. Published split names are retained.

## Hugging Face sources

Install `huggingface_hub`, `hf_xet`, and `pyarrow`. Downloads use pinned revisions,
reuse completed files in `--source-dir`, and use four download workers.
FLEURS uses the converted Parquet revision of `google/xtreme_s`, avoiding the
legacy Dataset loading script. ML-SUPERB uses `espnet/ml_superb_hf`.

```bash
export HF_XET_HIGH_PERFORMANCE=1
python -m egs3.voxlingua107.lid.src.fleurs_prepare \
  --source-dir /corpora/fleurs --output-dir egs3/voxlingua107/lid/data/fleurs \
  --languages ja_jp en_us --splits test
python -m egs3.voxlingua107.lid.src.ml_superb2_prepare \
  --source-dir /corpora/ml_superb2 --output-dir egs3/voxlingua107/lid/data/ml_superb2 \
  --splits dev dev_dialect
```

Omit `--languages` to prepare all FLEURS locales. FLEURS supports `train`,
`validation`, and `test`; ML-SUPERB supports `train`, `dev`, and `dev_dialect`.
ML-SUPERB retains the label aliases, Norwegian exclusion for train/dev, and
MS-Speech dialect-label correction from `egs2/ml_superb2/asr1/local/download.py`.
FLEURS locale tags are converted to LID codes; `nb_no` maps to `nor`, `fil_ph`
to `tgl`, and Chinese to `cmn`, consistent with the VoxLingua inventory.

## VoxPopuli

```bash
python -m egs3.voxlingua107.lid.src.voxpopuli_prepare \
  --source-dir /corpora/voxpopuli --output-dir egs3/voxlingua107/lid/data/voxpopuli \
  --languages en --splits test
```

Install `fsspec` with HTTP support (`aiohttp`). The script downloads Meta's ASR
annotations and the required original recordings from the official yearly tar
files. HTTP range requests skip unrelated recordings; this requires a server
and proxy supporting byte ranges. Completed recordings are reused after a
restart. Interrupted recordings are downloaded again. To use locally extracted
`raw_audios/original/<year>/<session>_original.ogg`, add `--skip-download-audio`.
Omitting `--languages` selects all 16 ASR languages. Splits are train/dev/test.

Segmentation follows `egs2/owsm_v3/s2t1/local/prepare_voxpopuli.py`: retain the
interval from the first VAD start to the last VAD end, including intervening
gaps, and omit empty-transcript or shorter-than-0.1-second entries.

## Babel (local LDC data)

```bash
python -m egs3.voxlingua107.lid.src.babel_prepare \
  --source-dir /corpora/babel/conversational/dev \
  --language asm --split dev --output-dir egs3/voxlingua107/lid/data/babel_asm
```

The raw source directory must contain `audio/` and `transcription/`. Babel is
not downloaded. SPHERE audio requires `sph2pipe` on PATH, or `--sph2pipe PATH`;
WAV sources are also accepted. Timestamp pairs and nonlexical-only segment
filtering follow `egs2/babel/asr1/local/prepare_acoustic_training_data.pl`.
The first audio channel is cut and resampled directly to 16 kHz. Supply the
language code and each raw split explicitly; no train/dev repartition is made.

## Evaluate or combine datasets

From the recipe directory, edit `prepared_name` and `prepared_manifest` in
`conf/inference_prepared.yaml`, then run the usual `infer measure` stages with
that inference config. Its Dataset selects only languages in the model's
training `lang2utt` and logs the retained count. No training audio is required.
Add separate `dataset.test` entries to evaluate multiple corpora independently.
For training, use the same `data_src` with a training manifest, omit the
`lang2utt` filter, and rerun `collect_stats` after combining datasets.

For a small preparation check, add `--max-utterances 3`. This limits each
FLEURS/VoxPopuli language and split, or each ML-SUPERB/Babel split. Hugging Face
checks fetch only the first matching shard. Omit this option for full
preparation; it is not a benchmark evaluation subset.
