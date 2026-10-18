# AmericasNLP 2022 ASR recipe

ESPnet3 port of [`egs2/americasnlp22/asr1`](../../../egs2/americasnlp22/asr1),
the baseline for the ASR task of
[the second AmericasNLP competition](http://turing.iimas.unam.mx/americasnlp/st.html).
It trains one **monolingual** model per language, exactly as the ESPnet2
recipe did:

| Code | Language | Archive |
|---|---|---|
| `bzd` | Bribri | `Bribri` |
| `gn` | Guarani | `Guarani` |
| `gvc` | Kotiria | `Kotiria` |
| `quy` | Quechua | `Quechua` |
| `tav` | Wa'ikhana | `Waikhana` |

Each model is an XLS-R 300M (wav2vec2) frontend, frozen except for its top two
transformer layers, followed by a linear preencoder (1024→80) and a 1-block
transformer encoder trained with pure CTC (`ctc_weight: 1.0`), with a
per-language SentencePiece **unigram** tokenizer (vocab 100). Validation
selects the checkpoint with the lowest `valid/cer_ctc`. The corpus has no test
split, so — as in ESPnet2 — the dev split is used for evaluation.

## Setup

```bash
cd egs3/americasnlp22/esp2_asr
. ./path.sh
```

The per-language corpus tarball is downloaded automatically from the shared
task server into `downloads/`. To reuse a pre-downloaded corpus instead, set
the `AMERICASNLP22` environment variable to its root (the directory holding
`Bribri/`, `Guarani/`, ...).

Note: the first `collect_stats`/`train`/`infer` run downloads the XLS-R
checkpoint (~1.2 GB) into `./hub`.

## Configs

`conf/training.yaml` and `conf/inference.yaml` hold everything the five
models share. The configs passed to `run.py` are the per-language
`conf/tuning/training_<lang>.yaml` and `conf/tuning/inference_<lang>.yaml`,
which only set `lang` on top of them (through `defaults:`). Every language
gets its own outputs: `data/<lang>_unigram100/` (tokenizer),
`exp/stats_<lang>/` (shape statistics) and `exp/training_<lang>/` (model and
inference results).

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

Results (WER/CER in percent) land in
`exp/training_tav/inference_tav/metrics.json`, next to the per-test-set
directory `dev_tav/` holding the hypotheses (`text.scp`), the references
(`dataset/text.scp`) and the WER/CER alignments. Replace `tav` with any other
language code for the remaining models; `conf/metrics.yaml` is shared across
languages.

## Results

Dev sets, scored by `measure` with the configs above; the sentence, word and
character counts match the ESPnet2 baseline below. Each model was trained on
one RTX 3090, shared by two or three of the language runs at a time (from
20 min for `gn` to 2 h 8 min for `gvc`), with PyTorch 2.14.1 (CUDA 13.0).

WER:

| dataset | Snt | Wrd | Corr | Sub | Del | Ins | Err | S.Err |
|---|---|---|---|---|---|---|---|---|
| dev_bzd | 250 | 2056 | 13.2 | 64.7 | 22.1 | 4.7 | 91.5 | 100.0 |
| dev_gn | 93 | 391 | 8.2 | 70.3 | 21.5 | 5.9 | 97.7 | 100.0 |
| dev_gvc | 253 | 2206 | 14.4 | 70.5 | 15.0 | 6.6 | 92.2 | 99.6 |
| dev_quy | 250 | 11465 | 19.1 | 72.7 | 8.2 | 6.0 | 86.9 | 100.0 |
| dev_tav | 250 | 1201 | 4.3 | 89.3 | 6.4 | 26.1 | 121.7 | 99.2 |

CER:

| dataset | Snt | Wrd | Corr | Sub | Del | Ins | Err | S.Err |
|---|---|---|---|---|---|---|---|---|
| dev_bzd | 250 | 10083 | 59.5 | 18.5 | 21.9 | 7.7 | 48.2 | 100.0 |
| dev_gn | 93 | 2946 | 78.5 | 10.5 | 11.0 | 6.6 | 28.1 | 100.0 |
| dev_gvc | 253 | 13453 | 65.6 | 15.8 | 18.6 | 8.4 | 42.8 | 99.6 |
| dev_quy | 250 | 95334 | 78.7 | 9.8 | 11.5 | 10.3 | 31.6 | 100.0 |
| dev_tav | 250 | 8606 | 56.4 | 27.1 | 16.5 | 13.2 | 56.8 | 99.2 |

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
  deterministic and aligned across `text.scp`/`dataset/text.scp`.
- **Speed perturbation** (`0.9/1.0/1.1`): ESPnet2 materializes three dataset
  copies before training (`train_<lang>_sp`) and trains 15 epochs over them;
  ESPnet3 applies one random factor per sample access via the preprocessor's
  `data_aug_effects`, so an epoch is a third of ESPnet2's. The recipe trains
  45 epochs and validates every third one: the same number of updates, the
  same 15 validations. Same factor set, different schedule, so training
  curves are not bit-identical.
- **Batch size**: ESPnet2's `batch_bins: 200000` counts raw 16 kHz samples,
  ESPnet3 counts the collected feature shapes (50 XLS-R frames per second ×
  1024 dims), so the same 12.5 s of audio per batch is `batch_bins: 640000`.
- **Duration filter** (`--max_wav_duration 38` and asr.sh's default
  `--min_wav_duration 0.1`): applied at dataset index time on raw durations
  to the training set and to the dev set, for validation and for scoring
  alike, as ESPnet2 decodes its filtered valid set (hence 253 of the 254
  Kotiria dev utterances). ESPnet2 filtered the training set after
  perturbation; the ≤1.1 speed factors make the difference negligible.
- **Frozen layers**: ESPnet2's `freeze_param` names
  (`frontend.upstream.model.*`) predate espnet2's `S3PRLUpstream` wrapper and
  match no parameter in current espnet2; the recipe freezes the same modules
  (the CNN feature extractor and transformer layers 0-21) under their current
  names, `frontend.upstream.upstream.model.*`.
- **Checkpoint selection**: the inference config references
  `valid.cer_ctc.ave_1best.pth`, the ESPnet3-averaged top-1 checkpoint —
  equivalent to ESPnet2's `valid.cer_ctc.best.pth`.
- **No LM, no text cleaner**: transcripts keep the corpus `source_raw` text
  verbatim, matching `cleaner=none` in ESPnet2.
- **Precision**: ESPnet3 trains with TF32 matrix multiplications
  (`torch.set_float32_matmul_precision("high")`), ESPnet2 in full FP32.
- WER above 100% (Wa'ikhana here; Guarani and Wa'ikhana in ESPnet2) is
  expected: the models are weak baselines on very small corpora and tend to
  over-generate.
