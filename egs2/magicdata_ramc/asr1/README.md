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
WARMSTART=1 ./run_e_branchformer.sh            # ASR warm-start from AISHELL
./run_e_branchformer_nolm.sh --stage 12 --stop-stage 13   # decode w/o LM
bash ./train_lm_warmstart.sh && \              # LM warm-start from magicdata
  ./run_decode_warmstart_lm.sh --stage 12 --stop-stage 13
```

Stage 1 downloads `MagicData-RAMC.tar.gz` from OpenSLR #123 (~6 GB) into
`${MAGICDATA_RAMC}`, extracts it, parses the per-conversation annotations into
per-utterance Kaldi-style data directories, and validates them. Stages 2-13 are
the usual ESPnet flow (speed-perturb, dump, token list, LM stats/train, ASR
stats/train, decode, score).

# E-Branchformer + Speed Perturbation + Transformer LM (best)

## Environments
- date: `Thu May 21 19:56:59 UTC 2026`
- python version: `3.10.20 (main, Mar 11 2026, 17:46:40) [GCC 14.3.0]`
- espnet2 version: `espnet2 202604`
- pytorch version: `pytorch 2.9.1+cu126`
- Git hash: `29a4df6492ff4ffef7b00e27eda9f4cffca22b4b`
  - Commit date: `Fri May 15 00:23:24 2026 +0900`

## With Transformer LM
- ASR config: [./conf/train_asr_e_branchformer.yaml](./conf/train_asr_e_branchformer.yaml)
- LM  config: [./conf/train_lm_transformer.yaml](./conf/train_lm_transformer.yaml)
- Decode config: [./conf/decode_asr_branchformer_lm.yaml](./conf/decode_asr_branchformer_lm.yaml)
- Speed-perturb factors: `0.9 1.0 1.1`
- max_epoch: 30 (ASR), 15 (LM)
- Trained from scratch on 2 × A800-80GB

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_branchformer_lm_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/dev|9790|130121|87.2|10.9|1.9|0.9|13.7|60.1|
|decode_asr_branchformer_lm_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/test|21216|273029|83.5|14.0|2.5|1.1|17.6|64.4|

### WER (whole-utterance error rate; char-segmented)

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_branchformer_lm_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/dev|9790|9790|39.9|60.1|0.0|0.0|60.1|60.1|
|decode_asr_branchformer_lm_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/test|21216|21216|35.6|64.4|0.0|0.0|64.4|64.4|



# E-Branchformer + SP, **no LM** (LM ablation)

## Environments
- date: `Fri May 22 07:38:46 UTC 2026`
- python version: `3.10.20 (main, Mar 11 2026, 17:46:40) [GCC 14.3.0]`
- espnet2 version: `espnet2 202604`
- pytorch version: `pytorch 2.9.1+cu126`
- Git hash: `29a4df6492ff4ffef7b00e27eda9f4cffca22b4b`

## Without LM
Same trained ASR model as the best config above (`exp/asr_e_branchformer_scratch`),
re-decoded with `--use_lm false` to isolate the LM rescoring contribution.
- ASR config: [./conf/train_asr_e_branchformer.yaml](./conf/train_asr_e_branchformer.yaml)
- Decode config: [./conf/decode_asr_branchformer.yaml](./conf/decode_asr_branchformer.yaml) (no LM)
- Driver: [./run_e_branchformer_nolm.sh](./run_e_branchformer_nolm.sh)

LM rescoring buys **dev −0.5 / test −0.4 CER**, almost entirely as
substitution reduction (Sub: 11.5→10.9 dev, 14.6→14.0 test). Deletion ticks
up marginally (+0.2) because LM nudges toward fluent omissions.

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_branchformer_asr_model_valid.acc.ave/dev|9790|130121|86.8|11.5|1.7|1.0|14.2|62.1|
|decode_asr_branchformer_asr_model_valid.acc.ave/test|21216|273029|83.2|14.6|2.3|1.2|18.0|66.0|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_branchformer_asr_model_valid.acc.ave/dev|9790|9790|37.9|62.1|0.0|0.0|62.1|62.1|
|decode_asr_branchformer_asr_model_valid.acc.ave/test|21216|21216|34.0|66.0|0.0|0.0|66.0|66.0|



# E-Branchformer + SP + LM (LM warm-started from magicdata)

## Environments
- date: `Fri May 22 09:53:35 UTC 2026`
- python version: `3.10.20 (main, Mar 11 2026, 17:46:40) [GCC 14.3.0]`
- espnet2 version: `espnet2 202604`
- pytorch version: `pytorch 2.9.1+cu126`
- Git hash: `29a4df6492ff4ffef7b00e27eda9f4cffca22b4b`

## With warm-started Transformer LM
Same trained ASR model as the best config above (`exp/asr_e_branchformer_scratch`),
re-decoded with a Transformer LM that was warm-started from
[espnet/jiyangtang_magicdata_asr_conformer_lm_transformer](https://huggingface.co/espnet/jiyangtang_magicdata_asr_conformer_lm_transformer)
instead of trained from random init. Same Transformer arch
(16 layers, 512 att, 2048 ffn, 128 embed); vocab differs 4486 vs 4035 so the
input embedding and the output projection are reinitialised via
`--ignore_init_mismatch true`.
- ASR config: [./conf/train_asr_e_branchformer.yaml](./conf/train_asr_e_branchformer.yaml)
- LM config: [./conf/train_lm_transformer_warmstart.yaml](./conf/train_lm_transformer_warmstart.yaml)
- Decode config: [./conf/decode_asr_branchformer_lm.yaml](./conf/decode_asr_branchformer_lm.yaml)
- LM driver: [./train_lm_warmstart.sh](./train_lm_warmstart.sh)
  (asr.sh stage 7 has no `--pretrained_lm` passthrough; this script invokes
  `espnet2.bin.lm_train` directly with `--init_param`)
- Decode driver: [./run_decode_warmstart_lm.sh](./run_decode_warmstart_lm.sh)
  (asr.sh `--lm_exp` points at the warm-started LM dir)
- LM training: 8 epochs fine-tune, lr=5e-4, warmup=5k (vs 15 ep / 1e-3 / 25k for scratch)

Note: this warm-start *underperforms* the from-scratch LM
(dev +0.3 CER, test +0.3 CER). The pretrained LM was trained on magicdata
**read-speech** text, whose character-distribution prior conflicts with the
**conversational** target domain. The warm-started LM's best valid loss
(3.831 at epoch 4) was worse than the scratch LM's best (3.754 at epoch 10),
which propagated through to a marginally worse rescoring outcome. This
section documents the recipe's ability to do LM warm-start; for production
use, the from-scratch LM remains the recommended choice on this corpus.

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_branchformer_lm_lm_lm_train_lm_transformer_zh_char_warmstart_valid.loss.ave_asr_model_valid.acc.ave/dev|9790|130121|86.9|11.2|1.9|0.9|14.0|60.7|
|decode_asr_branchformer_lm_lm_lm_train_lm_transformer_zh_char_warmstart_valid.loss.ave_asr_model_valid.acc.ave/test|21216|273029|83.2|14.3|2.5|1.1|17.9|64.9|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_branchformer_lm_lm_lm_train_lm_transformer_zh_char_warmstart_valid.loss.ave_asr_model_valid.acc.ave/dev|9790|9790|39.3|60.7|0.0|0.0|60.7|60.7|
|decode_asr_branchformer_lm_lm_lm_train_lm_transformer_zh_char_warmstart_valid.loss.ave_asr_model_valid.acc.ave/test|21216|21216|35.1|64.9|0.0|0.0|64.9|64.9|



# E-Branchformer + SP + LM, ASR encoder warm-started from AISHELL E-Branchformer

## Environments
- date: `Fri May 22 00:16:27 UTC 2026`
- python version: `3.10.20 (main, Mar 11 2026, 17:46:40) [GCC 14.3.0]`
- espnet2 version: `espnet2 202604`
- pytorch version: `pytorch 2.9.1+cu126`
- Git hash: `29a4df6492ff4ffef7b00e27eda9f4cffca22b4b`

## With Transformer LM
- ASR config: [./conf/train_asr_e_branchformer_warmstart.yaml](./conf/train_asr_e_branchformer_warmstart.yaml)
- LM  config: [./conf/train_lm_transformer.yaml](./conf/train_lm_transformer.yaml)
- Decode config: [./conf/decode_asr_branchformer_lm.yaml](./conf/decode_asr_branchformer_lm.yaml)
- Speed-perturb factors: `0.9 1.0 1.1`
- max_epoch: 20 (fine-tune schedule, lr=5e-4, warmup=10k)
- Pretrained encoder: [pyf98/aishell_e_branchformer](https://huggingface.co/pyf98/aishell_e_branchformer)
  (architecture-matched, loaded via `--pretrained_model` + `--ignore_init_mismatch true`;
  token-vocab-dependent layers reinitialised)
- Trained on 2 × A800-80GB

Note: this warm-start *underperforms* the scratch baseline above
(dev +3.5 CER, test +3.7 CER). AISHELL is read news speech, whose acoustic
prior actively conflicts with the conversational target domain. A
conversational-domain pretrained encoder (WenetSpeech-class) would be a more
sensible warm-start source; this config is kept here to document the recipe's
support for `--pretrained_model` rather than as a recommended setting.

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_branchformer_lm_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/dev|9790|130121|83.9|13.9|2.3|1.1|17.2|65.0|
|decode_asr_branchformer_lm_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/test|21216|273029|80.1|17.0|2.9|1.4|21.3|68.8|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_branchformer_lm_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/dev|9790|9790|35.0|65.0|0.0|0.0|65.0|65.0|
|decode_asr_branchformer_lm_lm_lm_train_lm_transformer_zh_char_valid.loss.ave_asr_model_valid.acc.ave/test|21216|21216|31.2|68.8|0.0|0.0|68.8|68.8|



# Branchformer-24

## Environments
- date: `Wed May 20 11:47:37 UTC 2026`
- python version: `3.10.20 (main, Mar 11 2026, 17:46:40) [GCC 14.3.0]`
- espnet2 version: `espnet2 202604`
- pytorch version: `pytorch 2.9.1+cu126`
- Git hash: `29a4df6492ff4ffef7b00e27eda9f4cffca22b4b`

## Without LM, without SP
- ASR config: [./conf/train_asr_branchformer.yaml](./conf/train_asr_branchformer.yaml)
- Decode config: [./conf/decode_asr_branchformer.yaml](./conf/decode_asr_branchformer.yaml)
- 24-block Branchformer, output_size=256, cgmlp_linear_units=2048
- max_epoch: 60
- Trained from scratch on 2 × A800-80GB

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_branchformer_asr_model_valid.acc.ave/dev|9790|130121|85.0|13.1|1.9|1.0|16.1|64.7|
|decode_asr_branchformer_asr_model_valid.acc.ave/test|21216|273029|81.3|16.2|2.5|1.2|19.9|68.4|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_branchformer_asr_model_valid.acc.ave/dev|9790|9790|35.3|64.7|0.0|0.0|64.7|64.7|
|decode_asr_branchformer_asr_model_valid.acc.ave/test|21216|21216|31.6|68.4|0.0|0.0|68.4|68.4|



# Conformer (initial baseline)

## Environments
- date: `Wed May 20 03:26:53 UTC 2026`
- python version: `3.10.20 (main, Mar 11 2026, 17:46:40) [GCC 14.3.0]`
- espnet2 version: `espnet2 202604`
- pytorch version: `pytorch 2.9.1+cu126`
- Git hash: `29a4df6492ff4ffef7b00e27eda9f4cffca22b4b`

## Without LM, without SP
- ASR config: [./conf/train_asr_conformer.yaml](./conf/train_asr_conformer.yaml)
- Decode config: [./conf/decode_asr_branchformer.yaml](./conf/decode_asr_branchformer.yaml)
- 12-block Conformer, output_size=256, hybrid CTC/attention (ctc_weight=0.3)
- max_epoch: 50
- Trained from scratch on 2 × A800-80GB

### CER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_branchformer_asr_model_valid.acc.ave/dev|9790|130121|84.7|13.5|1.8|1.2|16.5|66.0|
|decode_asr_branchformer_asr_model_valid.acc.ave/test|21216|273029|81.2|16.5|2.3|1.4|20.2|69.3|

### WER

|dataset|Snt|Wrd|Corr|Sub|Del|Ins|Err|S.Err|
|---|---|---|---|---|---|---|---|---|
|decode_asr_branchformer_asr_model_valid.acc.ave/dev|9790|9790|34.0|66.0|0.0|0.0|66.0|66.0|
|decode_asr_branchformer_asr_model_valid.acc.ave/test|21216|21216|30.7|69.3|0.0|0.0|69.3|69.3|



# Summary

| #  | Model                       | SP | LM         | ASR init     | dev CER | test CER |
|----|-----------------------------|---:|------------|--------------|--------:|---------:|
| 1  | Conformer-12                |  × | ×          | scratch      | 16.5    | 20.2     |
| 2  | Branchformer-24             |  × | ×          | scratch      | 16.1    | 19.9     |
| 3a | E-Branchformer-12 + SP      |  ✓ | ×          | scratch      | 14.2    | 18.0     |
| 3b | **E-Branchformer-12 + SP**  |  ✓ | **scratch**| scratch      | **13.7**| **17.6** |
| 3c | E-Branchformer-12 + SP      |  ✓ | magicdata-init | scratch  | 14.0    | 17.9     |
| 4  | E-Branchformer-12 + SP      |  ✓ | scratch    | AISHELL-init | 17.2    | 21.3     |

The recommended config is **(3b) E-Branchformer + SP + Transformer LM (both
trained from scratch)**.

## Ablation read-out

Holding the ASR model fixed (rows 3a / 3b / 3c — all use the same
`exp/asr_e_branchformer_scratch` model, only the decode-time LM differs):

- **LM rescoring contributes** ~0.4-0.5 absolute CER (3a → 3b), almost entirely
  from substitution reduction. A modest but real lever.
- **LM warm-start hurts** (3b → 3c, dev +0.3 / test +0.3): the magicdata
  read-speech LM's character-distribution prior conflicts with the
  conversational target. LM perplexity confirms: warm-start best valid loss =
  3.831 vs scratch best = 3.754.

Holding everything else fixed (rows 3b vs 4 — same ASR config, same SP, same
LM):

- **ASR encoder warm-start from AISHELL hurts more** (3b → 4, dev +3.5 /
  test +3.7): same domain-mismatch story at the encoder level, with much larger
  damage because the encoder is doing the heavy lifting.

Both warm-start failures point at the same lesson: **for conversational
RAMC, only conversational-domain pretrained checkpoints would be useful**
(e.g. WenetSpeech-class for the encoder; an in-domain conversational LM for
the LM). Read-speech pretraining is actively harmful here.
