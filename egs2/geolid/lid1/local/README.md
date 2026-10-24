# Data preparation

Run commands from `egs2/geolid/lid1`. The recipe entrypoint orchestrates corpus
preparation through the shared `lid.sh` pipeline:

```bash
./run.sh                         # VoxLingua107 only
./run.sh --setup combined        # All five corpora, then training/evaluation
```

## Dependencies and corpus paths

After setting up ESPnet's tools environment, install the recipe dependencies:

```bash
make -C ../../../tools s3prl.done transformers.done lang2vec.done
python3 -m pip install 'datasets>=2.16,<4' pycountry
# Required for Babel, including combined preparation
make -C ../../../tools sph2pipe
```

`local/path.sh` checks the Python dependencies and reports installation commands
for missing packages. The FLEURS loader uses dataset scripts, which were removed
in `datasets` 4. `lang2vec` prepares target tables; model execution reads those
tables without importing the package. The recipe path includes the installed
`sph2pipe` decoder.

Set `VOXLINGUA107` in `db.sh`. VoxLingua107 has an automatic download script;
FLEURS and ML-SUPERB2 are downloaded from Hugging Face. Combined preparation also
requires these manually downloaded corpora:

- **Babel:** obtain the licensed audio from [LDC](https://catalog.ldc.upenn.edu).
  The default layout is
  `downloads/babel/BABEL_<id>/conversational/{training,dev,reference_materials}`,
  with `lexicon.txt` under `reference_materials`.
- **VoxPopuli:** follow the [upstream download instructions](https://github.com/facebookresearch/voxpopuli).
  The default corpus root is `downloads/voxpopuli`; the adapter reads
  `_transcribed_data/<language>/asr_{train,dev}.tsv` and year-organized audio
  below that root. Its `test_voxpopuli_lang` split reuses the upstream dev set.

Pass alternate roots through the data-preparation options:

```bash
./run.sh --setup combined \
    --local_data_opts "--babel_path /corpora/babel --voxpopuli_path /corpora/voxpopuli"
```

The combined setup selects each corpus automatically. Its preparation helper,
`local/prepare_combined.sh`, uses these splits:

| Corpus | Training | Validation | Additional evaluation |
| --- | --- | --- | --- |
| VoxLingua107 | `train_voxlingua107_lang` | `dev_voxlingua107_lang` | — |
| FLEURS | `train_fleurs_lang` | `dev_fleurs_lang` | `test_fleurs_lang` |
| ML-SUPERB2 | `train_ml_superb2_lang` | `dev_ml_superb2_lang` | `dev_dialect_ml_superb2_lang` |
| Babel | `train_babel_lang` | `dev_babel_lang` | — |
| VoxPopuli | `train_voxpopuli_lang` | `dev_voxpopuli_lang` | `test_voxpopuli_lang` |

## Preparation and resuming

```bash
# Run only corpus preparation; no waveform filtering or combination yet
./run.sh --setup combined --stop_stage 1

# Materialize waveforms, filter Babel, and combine existing prepared corpora
./run.sh --setup combined --stage 3 --stop_stage 3

# Resume training/evaluation without preparing, filtering, or combining again
./run.sh --setup combined --stage 4 --stop_stage 7
```

Stages 1–3 run separately for each corpus using the standard pipeline. Stage 3
must finish for all corpora before `local/prepare_combined.sh` invokes
`local/filter_babel_over_10s.sh` and `local/combine.sh`. Babel filtering keeps
utterances of **at least 10 seconds** and regenerates matching language lists.
Combined preparation uses raw 16 kHz waveforms without speed perturbation.
Resume at `--stage 3` to skip stages 1 and 2 when the data directories already exist.

The combination script checks for duplicate utterance IDs and matching waveform,
language and duration keys while preserving source dumps. It generates
`lang2utt`, `category2utt`, `utt2dataset` and `dataset2utt` in
`dump/raw/train_all_no_filter_lang`, and creates
`data/combined_lang2vec.txt`. The target table omits `tok`, which has no URIEL geo
features; `tok` remains a classification label. Other missing features fail
preparation. Only combined training needs dataset balancing maps; validation
uses language balancing (`valid_batch_type: catpow`).

The experiment directories are `exp_voxlingua107` and `exp_combined`.
Use `--lid_config` to select another training configuration. Other shared pipeline
options are set explicitly in `run.sh` and `local/prepare_combined.sh`; edit
those arguments when changing dataset names or dump locations. See the
[recipe README](../README.md) for training examples and supported entrypoint options.

## Individual preparation and shared code

For troubleshooting, individual corpus preparation remains available:

```bash
bash local/prepare_babel.sh --dataset_path /corpora/babel
bash local/prepare_voxpopuli.sh --dataset_path /corpora/voxpopuli
bash local/prepare_fleurs.sh
bash local/prepare_ml_superb2.sh
bash local/prepare_voxlingua107.sh
```

To prepare an individual corpus through stage 3 with the shared pipeline:

```bash
./lid.sh --stage 1 --stop_stage 3 --nj 8 --feats_type raw --fs 16k \
    --expdir exp_combined --local_data_opts "--datasets fleurs" \
    --train_set train_fleurs_lang --valid_set dev_fleurs_lang \
    --test_sets test_fleurs_lang
```

- VoxLingua107 calls `egs2/voxlingua107/lid1/local/data.sh` and links its Python
  preparer. Its standard directories are copied to GeoLID's `_lang` names after
  preparation completes.
- FLEURS calls `egs2/fleurs/asr1/local/create_dataset.py`, including train,
  validation/dev and test splits. The local adapter converts its TSVs to
  language labels and waveform manifests.
- ML-SUPERB2 calls `egs2/ml_superb2/asr1/local/download.py`, reusing its language
  normalization and dialect correction; the adapter reads the resulting text
  labels and waveform paths.
- Babel calls `egs2/babel/asr1/local/prepare_acoustic_training_data.pl` for
  transcription segmentation and audio commands. The adapter adds language
  prefixes and `utt2lang`; merge outputs are truncated on reruns.

Stage 1 normally creates `data/voxlingua107_lang2vec.txt`. After preparing
VoxLingua107 directly instead, create its table before training:

```bash
python local/prepare_lang2vec.py \
    --lang2utt dump/raw/train_voxlingua107_lang/lang2utt \
    --output data/voxlingua107_lang2vec.txt
```
