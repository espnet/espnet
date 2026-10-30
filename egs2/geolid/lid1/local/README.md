# Corpus preparation

Run commands from `egs2/geolid/lid1` after the
[recipe installation](../README.md#installation).

## Corpus paths

Set `VOXLINGUA107` in `db.sh`; its preparer downloads the corpus if needed.
FLEURS and ML-SUPERB2 are downloaded from Hugging Face. Download these two
corpora separately before combined preparation:

- **Babel:** obtain licensed audio from [LDC](https://catalog.ldc.upenn.edu).
  Use `downloads/babel/BABEL_<id>/conversational/{training,dev,reference_materials}`,
  including `reference_materials/lexicon.txt`; install `sph2pipe` as above.
- **VoxPopuli:** follow the [upstream download instructions](https://github.com/facebookresearch/voxpopuli).
  Use `downloads/voxpopuli/transcribed_data/<language>/asr_{train,dev,test}.tsv`
  and year-organized audio below the corpus root. The legacy `_transcribed_data`
  directory is also accepted. Evaluation uses the official test partition.

Override corpus roots with:

```bash
./run.sh --setup combined \
    --local_data_opts "--babel_path /corpora/babel --voxpopuli_path /corpora/voxpopuli"
```

## Preparation steps

```bash
./run.sh --setup combined --stop_stage 3  # Prepare, dump, filter and combine
./run.sh --setup combined --stage 3 --stop_stage 3  # Resume from prepared data
```

The helper runs stages 1–3 for each corpus. After stage 3, it filters Babel to
at least 10 seconds and combines the raw 16 kHz training dumps into
`dump/raw/train_all_no_filter_lang`. It writes `data/combined_lang2vec.txt`,
omitting the unavailable `tok` geo target while retaining its classification label.

| Corpus | Training | Validation | Additional evaluation |
| --- | --- | --- | --- |
| VoxLingua107 | `train_voxlingua107_lang` | `dev_voxlingua107_lang` | — |
| FLEURS | `train_fleurs_lang` | `dev_fleurs_lang` | `test_fleurs_lang` |
| ML-SUPERB2 | `train_ml_superb2_lang` | `dev_ml_superb2_lang` | `dev_dialect_ml_superb2_lang` |
| Babel | `train_babel_lang` | `dev_babel_lang` | `dev_babel_over_10s_lang` (after filtering) |
| VoxPopuli | `train_voxpopuli_lang` | `dev_voxpopuli_lang` | `test_voxpopuli_lang` |

For individual corpus preparation:

```bash
bash local/prepare_voxlingua107.sh
bash local/prepare_fleurs.sh
bash local/prepare_ml_superb2.sh
bash local/prepare_babel.sh --dataset_path /corpora/babel
bash local/prepare_voxpopuli.sh --dataset_path /corpora/voxpopuli
```

To also materialize waveform dumps for one corpus, use the shared pipeline, e.g.:

```bash
./lid.sh --stage 1 --stop_stage 3 --nj 8 --feats_type raw --fs 16k \
    --expdir exp_combined --local_data_opts "--datasets fleurs" \
    --train_set train_fleurs_lang --valid_set dev_fleurs_lang \
    --test_sets test_fleurs_lang
```

Normal stage 1 also creates `data/voxlingua107_lang2vec.txt`. If preparing
VoxLingua107 directly, generate this training target table after dumping audio:

```bash
python local/prepare_lang2vec.py \
    --lang2utt dump/raw/train_voxlingua107_lang/lang2utt \
    --output data/voxlingua107_lang2vec.txt
```
