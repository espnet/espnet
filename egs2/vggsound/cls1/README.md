# VGGSound Audio Classification Recipe

This recipe fine-tunes a BEATs encoder for single-label audio classification on
VGGSound with 309 classes. Training uses `multi-class` classification.
Evaluation reports top-1 accuracy together with mean per-class AP/AUC computed
in a one-vs-rest manner from class posterior scores.

## Data Format

Set `VGGSOUND` in `db.sh` to the VGGSound root directory. The data preparation
script looks for metadata in one of these forms by default:

- `${VGGSOUND}/train.csv`, `${VGGSOUND}/valid.csv`, `${VGGSOUND}/test.csv`
- `${VGGSOUND}/train.tsv`, `${VGGSOUND}/valid.tsv`, `${VGGSOUND}/test.tsv`
- `${VGGSOUND}/metadata/train.csv`, `${VGGSOUND}/metadata/valid.csv`,
  `${VGGSOUND}/metadata/test.csv`

Headered metadata should contain a label column named `label`, `class`,
`category`, or `caption`. Audio paths can be provided with a column named
`audio`, `audio_path`, `wav`, `wav_path`, `path`, `filename`, or `file`.
If explicit audio paths are not present, the script tries common VGGSound names
from `video_id`/`youtube_id`/`ytid` plus `start`/`start_time`/`start_sec`.

Headerless CSV/TSV is interpreted as either:

```text
youtube_id,start_seconds,label
```

or:

```text
audio_path,label
```

The generated ESPnet data directories contain `wav.scp`, `text`, and `utt2spk`
for `train`, `valid`, and `test`.

## Checkpoint

Download a BEATs checkpoint and update `beats_ckpt_path` in
`conf/beats_cls_vggsound.yaml`.

## Run

First verify data preparation and class vocabulary:

```sh
./run.sh --stage 1 --stop_stage 4
```

Then train and evaluate:

```sh
./run.sh --stage 5 --stop_stage 8 --ngpu 1
```

You can pass explicit metadata files if your split files use different names:

```sh
./run.sh --stage 1 --stop_stage 4 \
    --local_data_opts "--train-metadata /path/train.csv --valid-metadata /path/valid.csv --test-metadata /path/test.csv"
```

## Metrics

Scoring follows the shared ESPnet classification scoring utility:

- `mean_acc`: top-1 accuracy computed with `argmax`
- `mAP`: mean average precision over 309 classes
- `mean_auc`: mean ROC-AUC over 309 classes

Although the task is single-label classification, `mAP` and `mean_auc` are
computed from one-hot targets and class posterior scores in a one-vs-rest
fashion for class-wise analysis.
