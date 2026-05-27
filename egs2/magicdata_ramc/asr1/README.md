# MagicData-RAMC: ASR Recipe

[MagicData-RAMC](https://www.openslr.org/123/) (Yang et al., INTERSPEECH 2022) is
a free 180-hour Mandarin **conversational** speech corpus, designed as a harder
benchmark than the usual read-speech datasets. Each recording is a ~30 min
two-party spontaneous dialogue, 16 kHz two-channel (one speaker per channel),
covering 15 everyday and semi-professional topics (daily life, food, sports,
technology, business, healthcare, etc.). The corpus comes with an official
train/dev/test split with disjoint speakers and topics.

After the recipe's standard filtering (segments outside 0.3–30 s or text outside
1–200 chars are dropped, paralinguistic tags such as `[LAUGHTER]`/`[MUSIC]` are
stripped from the text), the prepared Kaldi-style data directories are:

| split | sessions | speakers | utterances | hours |
|-------|---------:|---------:|-----------:|------:|
| train | 289      | 556      | 152,367    | 115.9 |
| dev   | 19       | 38       | 9,790      | 7.8   |
| test  | 43       | 86       | 21,216     | 16.0  |

Note: this is **not** the same as the existing `egs2/magicdata` recipe, which
covers the MAGICDATA Mandarin Chinese **Read Speech** corpus (OpenSLR #68) — a
separate, single-speaker read-aloud dataset.

## How to run

```bash
cd egs2/magicdata_ramc/asr1
# 1. point db.sh at a writable location with ~6 GB free; the corpus tarball
#    will be downloaded there automatically.
echo 'MAGICDATA_RAMC=/path/to/where/MagicData-RAMC/lives' >> db.sh
# 2. baseline run (Conformer, no SP, no LM):
./run.sh
# 3. best-known config (E-Branchformer + speed perturb + Transformer LM):
./run_e_branchformer.sh
# 4. variants (LM ablations / warm-start studies):
WARMSTART=1 ./run_e_branchformer.sh                          # ASR warm-start from AISHELL
./run_e_branchformer_nolm.sh --stage 12 --stop-stage 13      # decode w/o LM
bash ./train_lm_warmstart.sh && \                            # LM warm-start from magicdata
  ./run_decode_warmstart_lm.sh --stage 12 --stop-stage 13
```

Stage 1 downloads `MagicData-RAMC.tar.gz` from OpenSLR #123 (~6 GB) into
`${MAGICDATA_RAMC}`, extracts it, parses the per-conversation annotations into
per-utterance Kaldi-style data directories, and validates them. Stages 2-13 are
the usual ESPnet flow (speed-perturb, dump, token list, LM stats/train, ASR
stats/train, decode, score).

## Environments
- python version: `3.10.20 (main, Mar 11 2026, 17:46:40) [GCC 14.3.0]`
- espnet2 version: `espnet2 202604`
- pytorch version: `pytorch 2.9.1+cu126`
- Git hash: `29a4df6492ff4ffef7b00e27eda9f4cffca22b4b`
  (Commit date `Fri May 15 00:23:24 2026 +0900`)
- Hardware: 2 × A800-80GB

## Results

Six configurations were trained / decoded on the same data split. CER is the
primary metric (Mandarin char-level); the parenthesised numbers are the
substitution / deletion / insertion components, which the analysis below uses
to attribute where each setting wins or loses errors.

| #  | ASR encoder       | ASR init     | SP | LM | LM init        | ep (ASR / LM) | dev CER (Sub / Del / Ins) | test CER (Sub / Del / Ins) |
|----|-------------------|--------------|:--:|:--:|----------------|:-------------:|---------------------------|----------------------------|
| 1  | Conformer-12      | scratch      | ×  | ×  | —              | 50 / —        | 16.5  (13.5 / 1.8 / 1.2)  | 20.2  (16.5 / 2.3 / 1.4)   |
| 2  | Branchformer-24   | scratch      | ×  | ×  | —              | 60 / —        | 16.1  (13.1 / 1.9 / 1.0)  | 19.9  (16.2 / 2.5 / 1.2)   |
| 3a | E-Branchformer-12 | scratch      | ✓  | ×  | —              | 30 / —        | 14.2  (11.5 / 1.7 / 1.0)  | 18.0  (14.6 / 2.3 / 1.2)   |
| **3b ★** | **E-Branchformer-12** | **scratch** | **✓** | **✓** | **scratch**         | **30 / 15**   | **13.7  (10.9 / 1.9 / 0.9)** | **17.6  (14.0 / 2.5 / 1.1)** |
| 3c | E-Branchformer-12 | scratch      | ✓  | ✓  | magicdata-warm | 30 /  8       | 14.0  (11.2 / 1.9 / 0.9)  | 17.9  (14.3 / 2.5 / 1.1)   |
| 4  | E-Branchformer-12 | AISHELL-warm | ✓  | ✓  | scratch        | 20 / 15       | 17.2  (13.9 / 2.3 / 1.1)  | 21.3  (17.0 / 2.9 / 1.4)   |

Configs (linked from the table column "ASR encoder + ASR init"):
- Conformer-12: [conf/train_asr_conformer.yaml](./conf/train_asr_conformer.yaml)
- Branchformer-24: [conf/train_asr_branchformer.yaml](./conf/train_asr_branchformer.yaml)
- E-Branchformer-12 (scratch): [conf/train_asr_e_branchformer.yaml](./conf/train_asr_e_branchformer.yaml)
- E-Branchformer-12 (AISHELL warm-start): [conf/train_asr_e_branchformer_warmstart.yaml](./conf/train_asr_e_branchformer_warmstart.yaml) +
  pretrained [pyf98/aishell_e_branchformer](https://huggingface.co/pyf98/aishell_e_branchformer)
- LM (scratch): [conf/train_lm_transformer.yaml](./conf/train_lm_transformer.yaml)
- LM (magicdata warm-start): [conf/train_lm_transformer_warmstart.yaml](./conf/train_lm_transformer_warmstart.yaml) +
  pretrained [espnet/jiyangtang_magicdata_asr_conformer_lm_transformer](https://huggingface.co/espnet/jiyangtang_magicdata_asr_conformer_lm_transformer)
- Decode (LM rescoring): [conf/decode_asr_branchformer_lm.yaml](./conf/decode_asr_branchformer_lm.yaml)
- Decode (no LM): [conf/decode_asr_branchformer.yaml](./conf/decode_asr_branchformer.yaml)

WER columns (whole-utterance error rate at character-level segmentation) and
RTF / latency tables for each run are in the per-experiment
`exp/<asr_tag>/RESULTS.md` files.

## Why each setting performs the way it does

**Why row 2 (Branchformer-24) barely beats row 1 (Conformer-12) at this
data scale.** Going from 12 to 24 blocks and from Conformer to Branchformer
roughly triples encoder capacity, but with only ~116 h of training data this
extra capacity has almost nothing new to learn — the gain is just 0.3-0.4
CER. Architecture changes alone, without more data, hit a ceiling fast on
this corpus.

**Why row 3a (E-Branchformer-12 + SP) jumps 1.9-2.0 CER over the baseline.**
This is the biggest single lever in the table, and the gain comes from two
nearly-orthogonal contributions stacked together:
1. **Speed perturbation ×3** triples the effective amount of acoustic data
   the model sees per epoch. For an acoustic encoder trained from scratch on
   ~116 h, simply having 3× more samples to fit to is a large regularisation
   + data-augmentation boost.
2. **E-Branchformer (12 blocks)** is more parameter-efficient than the 24-block
   Branchformer of row 2: half the depth, comparable / better accuracy. So
   the comparison is more apples-to-apples than "smaller model is winning" —
   it's a better architecture combined with more effective data.

**Why row 3b (LM rescoring on) helps but only by 0.4-0.5 CER.** Inspecting
the (Sub / Del / Ins) breakdown shows the win is essentially all in the
substitution column (dev Sub 11.5 → 10.9, test 14.6 → 14.0). The LM is
picking better candidates among same-pronunciation Chinese characters in
n-best lists — exactly what an external LM is supposed to do — and barely
moves Del/Ins. The reason the gain is small is that the ASR model itself was
trained on 150k conversational utterances, so its implicit LM (decoder +
CTC) already captures most easy n-gram statistics; the external LM only
adds the marginal cases.

**Why row 3c (LM warm-start from magicdata) underperforms row 3b.** The
magicdata pretrained LM was trained on **read-speech** transcripts (the
vocab itself betrays this — characters like 歌 "song" appear in the top-10).
Conversational text in RAMC has very different character co-occurrence
statistics (more interjections, repetitions, colloquial particles, shorter
sentences). The pretrained Transformer layers carry a prior that conflicts
with the target distribution; the warm-started LM's best valid loss is
**3.831 at epoch 4**, worse than scratch's **3.754 at epoch 10**, and this
worse perplexity propagates straight into a worse rescoring outcome. Note
also that `--ignore_init_mismatch true` had to reinitialise the input
embedding (vocab 4486 vs 4035) and output projection — these are token-
vocab-dependent layers that can't be loaded, so a significant fraction of
the pretrained model is effectively cold-started anyway.

**Why row 4 (ASR encoder warm-start from AISHELL) is the worst of the
six — even worse than the Conformer baseline.** Same domain-mismatch story
as row 3c, but at the **encoder** level instead of LM, and the damage is
much larger (dev +3.5, test +3.7 vs row 3b). The encoder is doing the heavy
lifting in ASR: it has to map raw 16 kHz audio to character-aligned features.
AISHELL's acoustic conditions (clean read news in a quiet studio with
careful pronunciation) are essentially the *opposite* of RAMC's (spontaneous
telephone-style dialogue with overlap, hesitations, casual pronunciation).
The pretrained acoustic prior actively misleads the fine-tune. On top of
that, the fine-tune schedule (`max_epoch=20`, `lr=5e-4`) was chosen
assuming warm-start would converge faster — which it doesn't when it's
starting from the wrong place.

**Why scratch training wins in this recipe.** The same lesson appears
twice (rows 3c and 4): **read-speech pretraining is actively harmful for
conversational fine-tuning, because the source domain teaches the model the
wrong distribution.** With ~116 h of in-domain data, training a sized-to-fit
model (12-block E-Branchformer, ~38 M params) from scratch beats both
warm-start variants. To make warm-start a real win on this corpus would
require a **conversational-domain** pretrained model — e.g., WenetSpeech-
class for the encoder (10k+ hours of in-the-wild Mandarin), or an LM
trained on a real conversational corpus. Those are the only directions
likely to break the ~13.7 dev / 17.6 test CER plateau set by row 3b.

## TL;DR

Use **row 3b** (`./run_e_branchformer.sh`). On 2 × A800-80GB this takes
~3 h for the ASR run plus ~25 min decoding. SP gives the biggest gain, LM
rescoring gives a small free win, and both warm-start experiments (LM and
ASR) are kept in the recipe only as documentation of how `--init_param` /
`--pretrained_model` plumbing works in ESPnet — they should not be used on
this corpus with read-speech source checkpoints.
