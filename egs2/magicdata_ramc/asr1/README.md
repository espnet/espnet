# MagicData-RAMC: ASR Recipe

[MagicData-RAMC](https://www.openslr.org/123/) (Yang et al., INTERSPEECH 2022) is
a free 180-hour Mandarin **conversational** speech corpus, designed as a harder
benchmark than the usual read-speech datasets. Each recording is a ~30 min
two-party spontaneous dialogue, 16 kHz two-channel (one speaker per channel),
covering 15 everyday and semi-professional topics (daily life, food, sports,
technology, business, healthcare, etc.). The corpus comes with an official
train/dev/test split with disjoint speakers and topics.

Per ESPnet recipe convention, only the **train** split is filtered (segments
outside 0.3–30 s or with cleaned text outside 1–200 chars are dropped, and
purely-unintelligible `[*]` segments are removed). **Dev and test are kept as
released** so reported CER/WER reflects the full evaluation distribution.
Paralinguistic tags (`[+]`, `[*]`, `[LAUGHTER]`, `[SONANT]`, `[MUSIC]`) are
preserved as atomic tokens — `local/data.sh` writes them into
`data/nlsyms.txt` and `run.sh` forwards the file via `--nlsyms_txt` so the
stage-5 token-list builder keeps them as single tokens.

The prepared Kaldi-style data directories are:

| split | sessions | speakers | utterances | hours |
|-------|---------:|---------:|-----------:|------:|
| train | 289      | 556      | 165,733    | 125.3 |
| dev   | 19       | 38       | 10,440     | 8.2   |
| test  | 43       | 86       | 23,012     | 17.1  |

Note: this is **not** the same as the existing `egs2/magicdata` recipe, which
covers the MAGICDATA Mandarin Chinese **Read Speech** corpus (OpenSLR #68) — a
separate, single-speaker read-aloud dataset.

## How to run

```bash
cd egs2/magicdata_ramc/asr1
# 1. point db.sh at a writable location with ~6 GB free; the corpus tarball
#    will be downloaded there automatically.
echo 'MAGICDATA_RAMC=/path/to/where/MagicData-RAMC/lives' >> db.sh
# 2. default recipe = best config (row 3b: E-Branchformer + SP + Transformer LM, all from scratch):
./run.sh
```

Stage 1 downloads `MagicData-RAMC.tar.gz` from OpenSLR #123 (~6 GB) into
`${MAGICDATA_RAMC}`, extracts it, parses the per-conversation annotations into
per-utterance Kaldi-style data directories, and validates them. Stages 2-13 are
the usual ESPnet flow (speed-perturb, dump, token list, LM stats/train, ASR
stats/train, decode, score).

### Reproducing the other rows of the results table

The variants in the table below are all reached by overriding `run.sh`'s
defaults on the CLI (the trailing `"$@"` in `run.sh` is forwarded to `asr.sh`).
No additional driver scripts are needed.

```bash
# Row 1 — Conformer baseline, no SP, no LM
./run.sh \
    --asr_config conf/train_asr_conformer.yaml \
    --use_lm false \
    --inference_config conf/decode_asr_branchformer.yaml \
    --speed_perturb_factors ""

# Row 2 — Branchformer-24, no SP, no LM
./run.sh \
    --asr_config conf/train_asr_branchformer.yaml \
    --use_lm false \
    --inference_config conf/decode_asr_branchformer.yaml \
    --speed_perturb_factors ""

# Row 3a — E-Branchformer + SP, no LM (decode-only LM ablation against row 3b)
./run.sh \
    --use_lm false \
    --inference_config conf/decode_asr_branchformer.yaml \
    --stage 12 --stop-stage 13

# Row 4 — E-Branchformer + SP + LM, ASR encoder warm-started from AISHELL
# (download the pretrained ckpt to pretrained/aishell_e_branchformer.pth first)
./run.sh \
    --asr_config conf/train_asr_e_branchformer_warmstart.yaml \
    --pretrained_model pretrained/aishell_e_branchformer.pth \
    --ignore_init_mismatch true

# Row 3c — E-Branchformer + SP + LM, LM warm-started from magicdata
# (download the pretrained LM to pretrained/magicdata_lm.pth first)
bash local/train_lm_warmstart.sh                # produces exp/lm_train_lm_transformer_zh_char_warmstart
./run.sh \
    --lm_exp exp/lm_train_lm_transformer_zh_char_warmstart \
    --stage 12 --stop-stage 13
```

## Environments
- python version: `3.10.20 (main, Mar 11 2026, 17:46:40) [GCC 14.3.0]`
- espnet2 version: `espnet2 202604`
- pytorch version: `pytorch 2.9.1+cu126`
- Git hash: `29a4df6492ff4ffef7b00e27eda9f4cffca22b4b`
  (Commit date `Fri May 15 00:23:24 2026 +0900`)
- Hardware: 2 × A800-80GB

## Results

Recommended config (E-Branchformer-12 + speed perturbation + Transformer LM,
all from scratch on 2 × A800-80GB; ~30 ASR epochs + 15 LM epochs):

| dev CER (Sub / Del / Ins) | test CER (Sub / Del / Ins) |
|---------------------------|----------------------------|
| **14.0** (11.0 / 2.0 / 1.0) | **17.8** (14.0 / 2.6 / 1.2) |

CER is the primary metric (Mandarin char-level); the parenthesised numbers
are the substitution / deletion / insertion components.

The recipe also ships configs for several variants of this main pipeline
(Conformer / Branchformer-24 baselines, no-LM decoding, AISHELL ASR-encoder
warm-start, magicdata LM warm-start). Their CLI invocations are documented
in the "Reproducing the other rows of the results table" snippet above; the
configs are kept primarily as worked examples of ESPnet's `--asr_config` /
`--pretrained_model` / `--lm_exp` / `--nlsyms_txt` plumbing rather than as
production-recommended alternatives. Per-experiment scoring artefacts (CER
breakdown, WER, RTF, latency) land under `exp/<asr_tag>/RESULTS.md`.

## TL;DR

Use the default `./run.sh`. On 2 × A800-80GB the full pipeline takes ~7 h
end-to-end (data prep + LM train + ASR train + decode + score). Speed
perturbation gives the biggest single quality lever; external LM rescoring
gives a small additional win on top, almost entirely as substitution
reduction. The two warm-start variants in the recipe (ASR encoder from
AISHELL E-Branchformer, LM from magicdata read-speech LM) were tried and
found to underperform from-scratch on this corpus — both source models are
read-speech and their priors conflict with the conversational target
distribution. To make warm-start a real win here, a conversational-domain
pretrained model (WenetSpeech-class for the encoder, or an in-domain
conversational LM) would be needed.
