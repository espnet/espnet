# AmericasNLP 2022 ASR recipe (ESPnet3)

This is the ESPnet3 port of the baseline setup for the ASR task of
[the second AmericasNLP competition](http://turing.iimas.unam.mx/americasnlp/st.html)
(originally `egs2/americasnlp22`). It trains one **monolingual** model per
language, exactly as the ESPnet2 recipe did:

| Code | Language | Archive |
|---|---|---|
| `bzd` | Bribri | `Bribri` |
| `gn` | Guarani | `Guarani` |
| `gvc` | Kotiria | `Kotiria` |
| `quy` | Quechua | `Quechua` |
| `tav` | Wa'ikhana | `Waikhana` |

Each model is a frozen XLS-R 300M (wav2vec2) frontend with a 1-block
transformer encoder, a linear preencoder (1024→80), and pure CTC training
(`ctc_weight: 1.0`), with a per-language SentencePiece **unigram** tokenizer
(vocab 100). Validation selects the checkpoint with the lowest
`valid/cer_ctc`. The corpus has no test split, so — as in ESPnet2 — the dev
split is used for evaluation.

## Setup

```bash
cd egs3/americasnlp22/asr
source path.sh   # or `. path.sh`
```

The per-language corpus tarball is downloaded automatically from the shared
task server into `downloads/`. To reuse a pre-downloaded corpus instead, set
the `AMERICASNLP22` environment variable to its root (the directory holding
`Bribri/`, `Guarani/`, ...).

Note: the first `collect_stats`/`train`/`infer` run downloads the XLS-R
checkpoint (~1.2 GB) into `./hub`.

## Quick start (per language, e.g. Wa'ikhana `tav`)

```bash
# 1) Download and validate the corpus
python run.py --stages create_dataset \
    --training_config conf/tuning/training_tav.yaml

# 2) Train the tokenizer, collect feature shapes, train the model
python run.py --stages train_tokenizer collect_stats train \
    --training_config conf/tuning/training_tav.yaml

# 3) Run inference
python run.py --stages infer \
    --training_config conf/tuning/training_tav.yaml \
    --inference_config conf/tuning/inference_tav.yaml

# 4) Score
python run.py --stages measure \
    --training_config conf/tuning/training_tav.yaml \
    --inference_config conf/tuning/inference_tav.yaml \
    --metrics_config conf/metrics.yaml
```

Results (WER/CER in `percent`) land in
`exp/training_tav/inference_tav/dev_tav/metrics.json`; hypothesis/reference
SCP files are written next to it. Replace `tav` with any other language code
for the remaining models; `conf/metrics.yaml` is shared across languages.

## Pretrained models (ESPnet2 baselines)

- Bribri: https://huggingface.co/espnet/americasnlp22-asr-bzd
- Guarani: https://huggingface.co/espnet/americasnlp22-asr-gn
- Kotiria: https://huggingface.co/espnet/americasnlp22-asr-gvc
- Quechua: https://huggingface.co/espnet/americasnlp22-asr-quy
- Wa'ikhana: https://huggingface.co/espnet/americasnlp22-asr-tav

## Reference results (ESPnet2 baseline, dev sets)

WER:

| dataset | Snt | Wrd | Corr | Sub | Del | Ins | Err | S.Err |
|---|---|---|---|---|---|---|---|---|
| dev_bzd | 250 | 2056 | 15.3 | 65.1 | 19.6 | 7.5 | 92.3 | 100.0 |
| dev_gn | 93 | 391 | 11.5 | 73.7 | 14.8 | 12.5 | 101.0 | 100.0 |
| dev_gvc | 253 | 2206 | 12.4 | 72.4 | 15.1 | 6.7 | 94.2 | 99.6 |
| dev_quy | 250 | 11465 | 18.7 | 67.0 | 14.3 | 4.3 | 85.6 | 100.0 |
| dev_tav | 250 | 1201 | 3.0 | 83.1 | 13.9 | 17.0 | 114.0 | 99.6 |

CER:

| dataset | Snt | Wrd | Corr | Sub | Del | Ins | Err | S.Err |
|---|---|---|---|---|---|---|---|---|
| dev_bzd | 250 | 10083 | 64.0 | 15.1 | 20.9 | 9.2 | 45.2 | 100.0 |
| dev_gn | 93 | 2946 | 83.4 | 7.9 | 8.7 | 8.7 | 25.3 | 100.0 |
| dev_gvc | 253 | 13453 | 64.7 | 15.5 | 19.9 | 10.2 | 45.6 | 99.6 |
| dev_quy | 250 | 95334 | 78.6 | 8.0 | 13.4 | 10.1 | 31.5 | 100.0 |
| dev_tav | 250 | 8606 | 57.5 | 19.9 | 22.7 | 12.0 | 54.5 | 99.6 |

## Notes and differences vs the ESPnet2 recipe

- **SCP row ids**: samples deliberately carry no identifier field (the ESPnet3
  dataset contract returns only preprocessor-facing fields), so
  hypothesis/reference SCP files are keyed by corpus-order indices; the
  dataset sorts samples by their corpus utterance id, keeping rows
  deterministic and aligned across `hyp.scp`/`ref.scp`.
- **Speed perturbation** (`0.9/1.0/1.1`): ESPnet2 materializes three dataset
  copies before training (`train_<lang>_sp`); ESPnet3 applies one random
  factor per sample access via the preprocessor's `data_aug_effects`. Same
  factor set, different schedule, so training curves are not bit-identical.
- **Long-utterance filter** (`max_wav_duration 38`): applied at dataset index
  time on raw durations (ESPnet2 filtered after perturbation; the ≤1.1 speed
  factors make the difference negligible).
- **Checkpoint selection**: the inference config references
  `valid.cer_ctc.ave_1best.pth`, the ESPnet3-averaged top-1 checkpoint —
  equivalent to ESPnet2's `valid.cer_ctc.best.pth`.
- **No LM, no text cleaner**: transcripts keep the corpus `source_raw` text
  verbatim, matching `cleaner=none` in ESPnet2.
- WER above 100% (Guarani, Wa'ikhana) is expected here: the models are weak
  baselines on very small corpora and tend to over-generate.
