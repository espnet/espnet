# FLEURS / CS-FLEURS / CS-YODAS language-set generation (asr1)

Encoder-decoder counterpart of `../lid1`: the model generates the languages of
an utterance as a token sequence (`<eng>`, or `<ara> <eng>` for code-switched
speech). It uses [MMS-1B](https://huggingface.co/facebook/mms-1b)
(`facebook/mms-1b`, S3PRL frontend with learned layer weights), a 24-layer
Transformer encoder (d = 256) and a 4-layer Transformer decoder trained with the
attention loss only (`ctc_weight: 0`). The permutation-invariant loss
(`model: pit_att_espnet`) takes the lowest loss over the orders of the target
tokens (at most `model_conf.max_pit_tokens` of them), so the model is trained
on the language set rather than on an order. Data preparation is shared with
`../lid1` (`local/data.sh`); see its README for the corpora and the duration
policy. Only the scripts are shared: to reuse the lid1 downloads, set `FLEURS`,
`CS_FLEURS` and `CS_YODAS` in `db.sh` to absolute paths, or run
`ln -s ../lid1/downloads downloads`.

## Usage

Each system is trained in two phases: phase 1 keeps MMS-1B frozen (its layer
weights are trained) with lr 1e-3; phase 2 trains all parameters with lr 5e-6,
starting from the best phase-1 checkpoint with a new optimizer and warm-up.
`run.sh` defaults to the phase-1 config
(`conf/tuning/train_mms_transformer_pit_frozen.yaml`); phase 2 is a second run
from stage 11 with the phase-2 config, which is the released config
(`conf/train.yaml`):

```bash
# phase 1
./run.sh --stop_stage 11
# phase 2
./run.sh --stage 11 \
    --asr_config conf/tuning/train_mms_transformer_pit.yaml \
    --pretrained_model exp/asr_train_mms_transformer_pit_frozen_fleurs_csfleurs_raw/valid.loss.best.pth
```

`--data` (`fleurs`, `fleurs_csfleurs`, `fleurs_csfleurs_csyodas`, see
`../lid1`) selects the training data; the configs are set for
`fleurs_csfleurs`. For the other training data, pass a copy of the config with
the values below as `--asr_config`.

| Phase | Training data | `batch_size` | `accum_grad` | `max_epoch` | `num_iters_per_epoch` | `scheduler_conf.warmup_steps` | `best_model_criterion` |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| 1 | FLEURS | 8 | 4 | 30 | 3360 | 2520 | `valid.acc` (max), `valid.loss` (min) |
| 1 | FLEURS + CS-FLEURS + CS-YODAS | 4 | 8 | 30 | 8240 | 3090 | `valid.loss` (min) |
| 2 | FLEURS | 8 | 4 | 23 | 3360 | 2520 | `valid.acc` (max), `valid.loss` (min) |
| 2 | FLEURS + CS-FLEURS + CS-YODAS | 4 | 8 | 30 | 8240 | 3090 | `valid.loss` (min) |

`valid_batch_size` equals `batch_size`; all other values are those of the
config.

Stage 12 decodes with `conf/decode.yaml` (beam 5) and stage 13 is the scoring
of `asr.sh`; `run.sh` then computes the exact unordered language-set accuracy
with `local/score_lidseq.py`.

Each model was trained on one, two or four A6000 GPUs. `batch_size` is the
global batch that is split over the GPUs, so the effective batch size is
`batch_size * accum_grad` (32 utterances) for any `--ngpu`.
`num_iters_per_epoch` counts mini-batches: if you change `batch_size` and
`accum_grad` (keeping their product), scale `num_iters_per_epoch` inversely to
`batch_size` so that an epoch covers the same data; the scheduler steps count
parameter updates and stay the same.

## Models

All models are released under CC-BY-NC-4.0 (the MMS-1B backbone is CC-BY-NC-4.0).

| Training data | HF repo |
| --- | --- |
| FLEURS | [espnet/mms1b-lid-transformer-pit-fleurs](https://huggingface.co/espnet/mms1b-lid-transformer-pit-fleurs) |
| FLEURS + CS-FLEURS (`conf/train.yaml`) | [espnet/mms1b-lid-transformer-pit-fleurs-csfleurs](https://huggingface.co/espnet/mms1b-lid-transformer-pit-fleurs-csfleurs) |
| FLEURS + CS-FLEURS + CS-YODAS | [espnet/mms1b-lid-transformer-pit-fleurs-csfleurs-csyodas](https://huggingface.co/espnet/mms1b-lid-transformer-pit-fleurs-csfleurs-csyodas) |

## Results

Exact language-set accuracy (%) from the generated token sequence. "seen" /
"unseen": the two-language set does / does not occur in the CS-FLEURS training
data (the same split for every model). CS-FLEURS all is the union of the four
CS-FLEURS test sets.

| Test set | Split | n | FLEURS + CS-FLEURS (`conf/train.yaml`) | FLEURS + CS-FLEURS + CS-YODAS |
| --- | --- | ---: | ---: | ---: |
| FLEURS | all | 77,808 | 95.55 | 93.10 |
| CS-FLEURS all | all | 49,297 | 36.63 | 30.06 |
| CS-FLEURS all | seen | 20,154 | 73.49 | 73.51 |
| CS-FLEURS all | unseen | 29,143 | 11.13 | 0.01 |
| CS read | all | 5,818 | 0.00 | 0.22 |
| CS read | seen | 5,307 | 0.00 | 0.24 |
| CS read | unseen | 511 | 0.00 | 0.00 |
| CS XTTS1 | seen | 11,321 | 99.74 | 99.73 |
| CS XTTS2 | unseen | 17,226 | 0.00 | 0.00 |
| CS MMS | all | 14,932 | 45.31 | 23.53 |
| CS MMS | seen | 3,526 | 99.83 | 99.60 |
| CS MMS | unseen | 11,406 | 28.45 | 0.02 |

## Using a released model

With the decoding options of `conf/decode.yaml`:

```python
import soundfile
from espnet2.bin.asr_inference import Speech2Text

speech2text = Speech2Text.from_pretrained(
    "espnet/mms1b-lid-transformer-pit-fleurs-csfleurs",
    beam_size=5, ctc_weight=0.0, lm_weight=0.0, maxlenratio=0.05,
)
speech, rate = soundfile.read("speech.wav")  # 16 kHz
text, *_ = speech2text(speech)[0]  # e.g. "<ara> <eng>"
```

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

Please also cite FLEURS, CS-FLEURS, CS-YODAS and MMS when you use the data or
the models.
