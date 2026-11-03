# FLEURS / CS-FLEURS / CS-YODAS code-switched language identification (lid1)

Language identification for utterances that contain one or two languages. The
models use the pretrained [MMS-1B](https://huggingface.co/facebook/mms-1b)
(`facebook/mms-1b`, S3PRL frontend with learned layer weights) as the encoder
and [ECAPA-TDNN](https://arxiv.org/pdf/2005.07143) as the embedding extractor,
with three output heads:

| Head (`conf/tuning/train_mms_ecapa_<head>_*.yaml`) | Target | Loss |
| --- | --- | --- |
| `softmax_hard` | one class per language; a code-switched pair such as `ara-eng` is one atomic class | AAM-Softmax (sub-center) |
| `softmax_soft` | one-hot for one language, 0.5/0.5 for two languages | AAM-Softmax (sub-center) + KL divergence |
| `sigmoid` | one-hot or two-hot | AAM cosine logits (sub-center) + BCE, positive weight 50 |

The head and the training data are selected with `run.sh --head <head> --data
<data>`, where `<data>` is `fleurs` (102 languages), `fleurs_csfleurs` (+ 16
code-switched pairs from CS-FLEURS XTTS-train) or `fleurs_csfleurs_csyodas` (+
six base-language/English pairs from CS-YODAS). The config is
`conf/tuning/train_mms_ecapa_<head>_<data>.yaml`. The default (`sigmoid`,
`fleurs_csfleurs`) is also `conf/train.yaml`. The companion `../asr1` recipe trains encoder-decoder
models that generate the language set as a token sequence.

## Data

| Dataset | Use | Source |
| --- | --- | --- |
| [FLEURS](https://huggingface.co/datasets/google/fleurs) | train / valid / test, 102 languages, official splits | revision `70bb2e84b976b7e960aa89f1c648e09c59f894dd` |
| [CS-FLEURS](https://huggingface.co/datasets/byan/cs-fleurs) | `xtts/train` split 98:2 into train / valid by a hash of the utterance id; `read/test`, `xtts/test1`, `xtts/test2`, `mms/test` for evaluation | revision `0cdbf166c5517ae4b6eb1c54248522eedec53017` |
| [CS-YODAS](https://huggingface.co/datasets/byan/cs-yodas) | `ara`, `cmn`, `fra`, `hin`, `jpn`, `rus` + English records, split 80/10/10 per base language by video | revision `e51028041b403f63c99ac91a4af040e72d0cad0e` |

`local/data.sh` downloads the three corpora into `${FLEURS}/fleurs_cs_lid`,
`${CS_FLEURS}/cs-fleurs` and `${CS_YODAS}/cs-yodas` (see `db.sh`; about 300 GB for
FLEURS, 80 GB for CS-FLEURS and 130 GB for CS-YODAS including the extracted wav
files; all three are downloaded for every `--data`; `git-lfs` is required) and writes the data
directories listed in `local/prepare_fleurs_cs_lid_data.py`, named
`<split>_fleurscs_<view>[_pair]` (`view`: `fleurs`, `cs`, `csyodas` for
train/valid; `fleurs`, `cs_read`, `cs_xtts1`, `cs_xtts2`, `cs_mms` for test;
`_pair` views carry one atomic class per language set for the `softmax_hard`
models). Train and valid utterances keep `1 s <= duration < 30 s` (CS-YODAS:
`< 70 s`) by their metadata duration; test sets are not filtered.

## Usage

```bash
# default: sigmoid head, FLEURS + CS-FLEURS
./run.sh
# other systems
./run.sh --head softmax_hard --data fleurs_csfleurs_csyodas --ngpu 2
# a custom config
./run.sh --head sigmoid --data fleurs_csfleurs --lid_config conf/my_config.yaml
```

`run.sh` calls `lid.sh` with the training view, label file and feature type of
the selected system. Stage 7 of `lid.sh` runs `local/score.py`, which computes
the exact language-set accuracy from the stage-6 embeddings (top-1 for
FLEURS, top-2 for the code-switched sets, expanded atomic class for
`softmax_hard` models); `run.sh` then scores the union of the CS-FLEURS test
sets.

Each model was trained on one, two or four A6000 GPUs. `batch_size` is the
global batch that is split over the GPUs, so the effective batch size is
`batch_size * accum_grad` (32 utterances) for any `--ngpu`.
`num_iters_per_epoch` counts mini-batches: if you change `batch_size` and
`accum_grad` (keeping their product), scale `num_iters_per_epoch` inversely to
`batch_size` so that an epoch covers the same data; the scheduler steps count
parameter updates and stay the same.

## Models

All models are released under CC-BY-NC-4.0 (the MMS-1B backbone is CC-BY-NC-4.0).

| Config (`conf/tuning/`) | Data | HF repo |
| --- | --- | --- |
| `train_mms_ecapa_softmax_hard_fleurs.yaml` | FLEURS | [shun3232/mms1b-lid-softmax-hard-fleurs](https://huggingface.co/shun3232/mms1b-lid-softmax-hard-fleurs) |
| `train_mms_ecapa_softmax_hard_fleurs_csfleurs.yaml` | FLEURS + CS-FLEURS | [shun3232/mms1b-lid-softmax-hard-fleurs-csfleurs](https://huggingface.co/shun3232/mms1b-lid-softmax-hard-fleurs-csfleurs) |
| `train_mms_ecapa_softmax_hard_fleurs_csfleurs_csyodas.yaml` | FLEURS + CS-FLEURS + CS-YODAS | [shun3232/mms1b-lid-softmax-hard-fleurs-csfleurs-csyodas](https://huggingface.co/shun3232/mms1b-lid-softmax-hard-fleurs-csfleurs-csyodas) |
| `train_mms_ecapa_softmax_soft_fleurs_csfleurs.yaml` | FLEURS + CS-FLEURS | [shun3232/mms1b-lid-softmax-soft-fleurs-csfleurs](https://huggingface.co/shun3232/mms1b-lid-softmax-soft-fleurs-csfleurs) |
| `train_mms_ecapa_softmax_soft_fleurs_csfleurs_csyodas.yaml` | FLEURS + CS-FLEURS + CS-YODAS | [shun3232/mms1b-lid-softmax-soft-fleurs-csfleurs-csyodas](https://huggingface.co/shun3232/mms1b-lid-softmax-soft-fleurs-csfleurs-csyodas) |
| `train_mms_ecapa_sigmoid_fleurs.yaml` | FLEURS | [shun3232/mms1b-lid-sigmoid-fleurs](https://huggingface.co/shun3232/mms1b-lid-sigmoid-fleurs) |
| `train_mms_ecapa_sigmoid_fleurs_csfleurs.yaml` (`conf/train.yaml`) | FLEURS + CS-FLEURS | [shun3232/mms1b-lid-sigmoid-fleurs-csfleurs](https://huggingface.co/shun3232/mms1b-lid-sigmoid-fleurs-csfleurs) |
| `train_mms_ecapa_sigmoid_fleurs_csfleurs_csyodas.yaml` | FLEURS + CS-FLEURS + CS-YODAS | [shun3232/mms1b-lid-sigmoid-fleurs-csfleurs-csyodas](https://huggingface.co/shun3232/mms1b-lid-sigmoid-fleurs-csfleurs-csyodas) |

## Results

Exact language-set accuracy (%). "seen" / "unseen": the two-language set does /
does not occur in the CS-FLEURS training data (the same split for every model).
CS-FLEURS all is the union of the four CS-FLEURS test sets.

### `softmax_hard` models

| Test set | Split | n | FLEURS | FLEURS + CS-FLEURS | FLEURS + CS-FLEURS + CS-YODAS |
| --- | --- | ---: | ---: | ---: | ---: |
| FLEURS | all | 77,808 | 96.71 | 97.30 | 97.38 |
| CS-FLEURS all | all | 49,297 | 8.33 | 30.04 | 30.12 |
| CS-FLEURS all | seen | 20,154 | 10.70 | 73.47 | 73.66 |
| CS-FLEURS all | unseen | 29,143 | 6.69 | 0.00 | 0.00 |
| CS read | all | 5,818 | 13.42 | 0.19 | 0.69 |
| CS read | seen | 5,307 | 14.60 | 0.21 | 0.75 |
| CS read | unseen | 511 | 1.17 | 0.00 | 0.00 |
| CS XTTS1 | seen | 11,321 | 1.71 | 99.58 | 99.68 |
| CS XTTS2 | unseen | 17,226 | 0.51 | 0.00 | 0.00 |
| CS MMS | all | 14,932 | 20.37 | 23.59 | 23.58 |
| CS MMS | seen | 3,526 | 33.66 | 99.89 | 99.86 |
| CS MMS | unseen | 11,406 | 16.26 | 0.00 | 0.00 |

### `softmax_soft` and `sigmoid` models

| Test set | Split | n | `softmax_soft` FLEURS + CS-FLEURS | `softmax_soft` FLEURS + CS-FLEURS + CS-YODAS | `sigmoid` FLEURS + CS-FLEURS | `sigmoid` FLEURS + CS-FLEURS + CS-YODAS |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| FLEURS | all | 77,808 | 91.35 | 88.23 | 96.50 | 93.62 |
| CS-FLEURS all | all | 49,297 | 52.04 | 32.97 | 59.71 | 58.72 |
| CS-FLEURS all | seen | 20,154 | 95.20 | 59.65 | 96.64 | 96.77 |
| CS-FLEURS all | unseen | 29,143 | 22.20 | 14.51 | 34.17 | 32.41 |
| CS read | all | 5,818 | 75.52 | 5.28 | 80.49 | 82.47 |
| CS read | seen | 5,307 | 82.65 | 5.69 | 88.13 | 88.58 |
| CS read | unseen | 511 | 1.57 | 0.98 | 1.17 | 18.98 |
| CS XTTS1 | seen | 11,321 | 99.65 | 77.21 | 99.61 | 99.65 |
| CS XTTS2 | unseen | 17,226 | 0.00 | 0.01 | 0.00 | 0.00 |
| CS MMS | all | 14,932 | 66.84 | 48.23 | 90.24 | 86.19 |
| CS MMS | seen | 3,526 | 99.80 | 84.49 | 99.89 | 99.89 |
| CS MMS | unseen | 11,406 | 56.65 | 37.02 | 87.25 | 81.96 |

**Note:** the models were trained with `datasets` 4.8.5 and s3prl 0.4.18.

## Using a released model

Each HF repository is the model packed by `lid.sh` stage 9, in the layout of
this recipe: `exp/lid_<config name>_<feats_type>/` (`config.yaml` and
the checkpoint) and the class list
`dump/<feats_type>/<train_set>/lang2utt` read by the config. Language ids and
per-utterance embeddings of a data directory, with the repository cloned in this
recipe directory and run from its root:

```bash
git clone https://huggingface.co/shun3232/mms1b-lid-sigmoid-fleurs-csfleurs  # needs git-lfs
cd mms1b-lid-sigmoid-fleurs-csfleurs
exp=exp/lid_train_mms_ecapa_sigmoid_fleurs_csfleurs_raw
out=${exp}/inference/30epoch/test_fleurscs_cs_read
mkdir -p ${out}
python -m espnet2.bin.lid_inference --ngpu 1 --output_dir ${out} \
    --data_path_and_name_and_type ../dump/raw/test_fleurscs_cs_read/wav.scp,speech,sound \
    --lid_train_config ${exp}/config.yaml --lid_model_file ${exp}/30epoch.pth \
    --use_preprocessor false --fix_duration false --extract_embd true \
    --save_embd_per_utt true --save_embd_avg_lang false --save_tsne_plot false
```

To score a released model as in the tables, prepare the data
(`./run.sh --stop_stage 3`), copy its `exp/lid_*` directory into `exp/` and run
`./run.sh --stage 6 --stop_stage 7` with the matching `--head`, `--data` and
`--inference_model <checkpoint>`.

## Citation

```bibtex
@misc{mitsumori2026cslid,
  title         = {Code-Switching Spoken Language Identification as Multi-Label Set Prediction},
  author        = {Mitsumori, Shunsuke and Wiesner, Matthew and Morishima, Shigeo and Watanabe, Shinji},
  year          = {2026},
  eprint        = {2610.01450},
  archivePrefix = {arXiv},
  primaryClass  = {eess.AS}
}
```

Please also cite FLEURS, CS-FLEURS, CS-YODAS and MMS (linked above) when you
use the data or the models.
