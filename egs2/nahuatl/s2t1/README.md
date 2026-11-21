# Nahuatl ASR (OWSM v4 fine-tuning)

Fine-tunes [`espnet/owsm_v4_medium_1B`](https://huggingface.co/espnet/owsm_v4_medium_1B)
for Nahuatl, an indigenous Mexican language, across three regional dialects:
**Hidalgo**, **Orizaba-Zongolica**, and **Zacatlan-Tepetzintla**.

## Dialect prompt-conditioning

OWSM has a language token for Nahuatl (`<na>`) but no per-dialect tokens. Instead
of adding new vocabulary (which would require patching the checkpoint's embedding
table, the BPE model, and the token list), this recipe distinguishes the three
dialects with **prompt-conditioning**:

- `text` uses OWSM's existing `<na>` language slot:
  `<na><asr><notimestamps> <transcript>`
- `text.prev` carries the dialect as a natural-language prompt:
  `nahuatl hidalgo` / `nahuatl orizaba zongolica` / `nahuatl zacatlan tepetzintla`

Fine-tuning therefore starts directly from the **released** OWSM checkpoint and
reuses its BPE model and token list unchanged. On our test sets this matches a
dedicated-vocabulary-token variant within ~0.5% CER while removing all checkpoint
surgery.

At decode time the same prompt is fed to the decoder (see `local/decode.py`); the
language symbol is always `<na>`, so all three dialects can be decoded in one
pass and are reported both per dialect and combined.

## Prerequisites

1. **OWSM checkpoint.** Download `espnet/owsm_v4_medium_1B` from HuggingFace into
   `${MODEL_CACHE_DIR}/owsm_v4_medium_1B/` (`MODEL_CACHE_DIR` is set in `path.sh`
   and defaults to `<repo_root>/../model_cache`). The download must include:
   - `exp/s2t_train_conv2d8_size1024_e18_d18_mel128_raw_bpe50000/`
     (`valid.total_count.ave_5best.pth`, `config.yaml`,
     `s2t_stats_raw_bpe50000/train/feats_stats.npz`)
   - `data/token_list/bpe_unigram50000/bpe.model`

2. **HuggingFace dataset.** Build the Nahuatl dataset (audio + time-aligned
   transcripts, speaker-disjoint splits per dialect) and set its path in `db.sh`
   as the `NAHUATL` corpus entry (like every other corpus location in `egs2/`).
   The dataset exposes nine splits:
   `{hidalgo,orizaba-zongolica,zacatlan-tepetzintla}-{train,val,test}`.

## Usage

```bash
# 1. Data prep: HF splits -> Kaldi data dirs, merged by set
./local/data.sh

# 2. Collect stats + fine-tune (from the released OWSM checkpoint)
./run.sh                              # data prep through training
./run.sh --stage 11 --stop_stage 11  # train only (stage 10 already done)

# 3. Decode + score the three test sets (per-dialect + combined CER)
./run.sh --stage 12 --stop_stage 13
```

Fine-tuning reuses OWSM's global-MVN feature statistics: `run.sh` runs
`collect_stats` (stage 10) to produce the per-utterance shape files, overwrites
the resulting `feats_stats.npz` with OWSM's, then trains (stage 11).

`cmd.sh` uses `run.pl` (local execution). On a cluster, set the backend and
scheduler options as usual for your site; the recipe carries no site-specific
scheduler settings.

## Results

`valid.acc.ave.pth`, beam size 5, CTC weight 0.3. Character error rate (%),
scored symmetrically (special tokens stripped from both hypothesis and
reference):

| Test set | CER |
|---|---|
| Hidalgo | 27.0 |
| Orizaba-Zongolica | 29.7 |
| Zacatlan-Tepetzintla | 18.3 |
| **Combined** | **24.9** |

## Files

| Path | Purpose |
|---|---|
| `run.sh` | Top-level driver: token list + stats setup, delegate to `s2t.sh`, decode |
| `local/data.sh` | HF splits → Kaldi data dirs, merge by set |
| `local/data_prep.py` | One HF split → a Kaldi data dir (writes the dialect prompt) |
| `local/decode.py` | Prompt-conditioned decoding + CER scoring |
| `conf/tuning/train_owsm_v4_nahuatl.yaml` | Fine-tuning config (`conf/train.yaml` → this) |
| `conf/decode.yaml` | Decoding config (beam size, CTC weight) |
