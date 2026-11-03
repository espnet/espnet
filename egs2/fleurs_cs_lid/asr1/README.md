# FLEURS / CS-FLEURS / CS-YODAS language-set generation (asr1)

Encoder-decoder counterpart of `../lid1`: the model generates the languages of
an utterance as a token sequence (`<eng>`, or `<ara> <eng>` for code-switched
speech). It uses [MMS-1B](https://huggingface.co/facebook/mms-1b)
(`facebook/mms-1b`, S3PRL frontend with learned layer weights), a 24-layer
Transformer encoder (d = 256) and a 4-layer Transformer decoder trained with the
attention loss only (`ctc_weight: 0`). For two-language targets the
permutation-invariant loss (`model_conf.pit_loss: true`) takes the lower of the
two token orders, so the model is trained on the language set rather than on an
order. Data preparation is shared with `../lid1` (`local/data.sh`); see its
README for the corpora and the duration policy. Only the scripts are shared:
to reuse the lid1 downloads, set `FLEURS`, `CS_FLEURS` and `CS_YODAS` in
`db.sh` to absolute paths, or run `ln -s ../lid1/downloads downloads`.

## Usage

Each system is trained in two phases: phase 1 keeps MMS-1B frozen (its layer
weights are trained) with lr 1e-3; phase 2 trains all parameters with lr 5e-6,
starting from the best phase-1 checkpoint with a new optimizer and warm-up.
`run.sh` defaults to the phase-1 config
(`conf/tuning/train_mms_transformer[_pit]_<data>_frozen.yaml`); phase 2 is a
second run from stage 11 with the phase-2 config, which is the released config
(`conf/train.yaml` links to the default one). For the default data:

```bash
# phase 1
./run.sh --data fleurs_csfleurs --stop_stage 11
# phase 2
./run.sh --data fleurs_csfleurs --stage 11 \
    --asr_config conf/tuning/train_mms_transformer_pit_fleurs_csfleurs.yaml \
    --pretrained_model exp/asr_train_mms_transformer_pit_fleurs_csfleurs_frozen_raw/valid.loss.best.pth
```

The same pattern holds for `--data fleurs` (`train_mms_transformer_fleurs*`)
and `--data fleurs_csfleurs_csyodas` (whose experiment directories end in
`_raw_copy` instead of `_raw`).

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
With single-language targets (FLEURS only) the permutation-invariant loss
reduces to the ordinary cross entropy, so that config does not set it.

| Config (`conf/tuning/`) | Data | HF repo |
| --- | --- | --- |
| `train_mms_transformer_fleurs.yaml` | FLEURS | [shun3232/mms1b-lid-transformer-pit-fleurs](https://huggingface.co/shun3232/mms1b-lid-transformer-pit-fleurs) |
| `train_mms_transformer_pit_fleurs_csfleurs.yaml` (`conf/train.yaml`) | FLEURS + CS-FLEURS | [shun3232/mms1b-lid-transformer-pit-fleurs-csfleurs](https://huggingface.co/shun3232/mms1b-lid-transformer-pit-fleurs-csfleurs) |
| `train_mms_transformer_pit_fleurs_csfleurs_csyodas.yaml` | FLEURS + CS-FLEURS + CS-YODAS | [shun3232/mms1b-lid-transformer-pit-fleurs-csfleurs-csyodas](https://huggingface.co/shun3232/mms1b-lid-transformer-pit-fleurs-csfleurs-csyodas) |

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

Each HF repository is the model packed by `asr.sh` stage 14, in the layout of
this recipe: `exp/asr_<config name>_<feats_type>/` holds `config.yaml` (with the
token list) and the checkpoint. From the root of the
cloned repository, with the decoding options of `conf/decode.yaml`:

```bash
git clone https://huggingface.co/shun3232/mms1b-lid-transformer-pit-fleurs-csfleurs  # needs git-lfs
cd mms1b-lid-transformer-pit-fleurs-csfleurs
```

```python
import soundfile
from espnet2.bin.asr_inference import Speech2Text

exp = "exp/asr_train_mms_transformer_pit_fleurs_csfleurs_raw"
speech2text = Speech2Text(
    asr_train_config=f"{exp}/config.yaml", asr_model_file=f"{exp}/15epoch.pth",
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
