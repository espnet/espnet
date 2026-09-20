# must_c st data recipe

This egs3 data module reads the **raw MuST-C v1** corpus release directly —
no Kaldi data preparation (`egs2/must_c/st1/local/data.sh`) is required.

- Language pair: the training config pins **en-de** (English speech, German
  translation), matching `src_lang`/`tgt_lang` in `egs2/must_c/st1/run.sh`.
  The builder itself defaults to `tgt_lang: all` (`dataset/config.yaml`) and
  will index every `en-<tgt>` pair present under the corpus root, so pass
  `tgt_lang="de"` — as `conf/tuning/train_st_conformer.yaml` does — to
  reproduce egs2 exactly.
- Corpus version: **v1** (`MUSTC_v1.0_en-de.tar.gz` from
  https://mt.fbk.eu/must-c-release-v1-0/), matching the `"v1"` argument
  `egs2/must_c/st1/local/data.sh` passes to `local/data_prep.sh`.
- Expected on-disk layout (unpacked corpus root, containing the `en-de/`
  language-pair directory):

  ```
  <corpus_root>/en-de/data/<split>/txt/<split>.yaml   # duration/offset/speaker_id/wav per segment
  <corpus_root>/en-de/data/<split>/txt/<split>.en     # English transcript, one line per segment
  <corpus_root>/en-de/data/<split>/txt/<split>.de     # German translation, one line per segment
  <corpus_root>/en-de/data/<split>/wav/<talk>.wav     # per-talk audio, sliced by offset/duration
  ```

  where `<split>` is one of `train`, `dev`, `tst-COMMON`, `tst-HE`.

- Source root resolution order: an explicit `source_dir=` argument, then the
  `MUST_C` environment variable, then the default path
  `/work/hdd/bbjs/shared/corpora/must-c_v1.2` (the on-disk location of this
  corpus release on this cluster). Note this default path's directory name
  differs from the `MUST_C` variable name used by the egs2 recipe/`db.sh` —
  either set `MUST_C` to point elsewhere, or rely on the default.

- Logical `test` maps to the physical `tst-COMMON` split (`split_aliases`),
  matching the egs2 recipe's convention. `tst-HE` is available as its own
  split (`"tst-HE"`).

- Each example returns three keys: `speech` (float32 waveform, native MuST-C
  sample rate), `text` (the target text selected by `task`, default `"st"` →
  target-language translation; `task="asr"` → English transcript), and
  `src_text` (English). Passing `return_utt_id=True` adds `utt_id`, which
  `conf/inference.yaml` does for its test entries.

  Training cannot carry `utt_id`: `CommonCollateFn` pads and stacks every value
  in the sample dict, so a `str` raises `AttributeError: 'str' object has no
  attribute 'dtype'` on the first batch of train and of `collect_stats` alike.
  Inference reads samples one at a time without collating, and requires an
  identifier (`InferenceRunner.idx_key` defaults to `utt_id`).

- **Case conventions from `run.sh` ARE reproduced**, because they change which
  characters exist at all: `src_case="lc.rm"` (lowercased, punctuation
  stripped, apostrophes kept) and `tgt_case="tc"` (truecase). What is *not*
  reproduced from egs2 is Moses `normalize-punctuation.perl` /
  `tokenizer.perl`; SentencePiece is trained on the cased-but-untokenized text
  and does its own segmentation.

- **Long/short filtering** reproduces `st.sh` stage 4: train and dev keep only
  segments strictly between 0.1 s and 20 s, test sets are left whole. Bounds and
  the split list live under `filter:` in `dataset/config.yaml`; the logic is
  `keep_duration()` / `kept_indices()` in `dataset/builder.py`. Counts:

  | split | as released | after filter |
  |---|---|---|
  | train | 229,703 | **220,466** |
  | dev | 1,423 | **1,381** |
  | test (tst-COMMON) | 2,641 | 2,641 |
  | tst-HE | 600 | 600 |

  The HF cache stays unfiltered — it is the analogue of st.sh's
  `dump/raw/org/<dset>`, and the Dataset applies the bounds when reading it, so
  changing them needs no cache rebuild. Pass `apply_filter=False` to see the
  corpus as released. The SentencePiece text is deliberately **not** filtered:
  `egs2/must_c/st1/run.sh:48-49` overrides `src_bpe_train_text`/
  `tgt_bpe_train_text` to the unfiltered `data/${train_set}/` rather than
  letting `st.sh` default them to the filtered dump, so egs2 builds its
  vocabularies over all 229,703 utterances while training on 220,466.

- Scope: this module reproduces the **egs2 ST recipe** (`egs2/must_c/st1`).

## Results

MuST-C en-de, conformer ST (`conf/tuning/train_st_conformer.yaml`), decoded with
`conf/inference.yaml` (beam 10) and scored with `conf/metrics.yaml`
(sacreBLEU, `tok:13a`). This is both passes `st.sh` runs: case-sensitive, and
case-insensitive after `remove_punctuation.pl` (suffix `_lc`). All twelve
numbers were checked against the egs2 shell pipeline itself and match exactly.

Converged model: `valid.acc.ave_10best`, the average of the ten best-`valid.acc`
checkpoints (epochs 53-74) after the full 80 epochs. Lower is better for TER.

| test set | utts | BLEU | chrF2 | TER | BLEU_lc | chrF2_lc | TER_lc | 1/2/3/4-gram precision | BP |
|---|---|---|---|---|---|---|---|---|---|
| tst-COMMON | 2,641 | **24.22** | 50.76 | 61.80 | 23.54 | 51.44 | 57.85 | 61.0 / 33.2 / 20.5 / 13.3 | 0.889 |
| tst-HE | 600 | **22.85** | 49.35 | 66.16 | 21.07 | 49.80 | 62.76 | 58.8 / 31.9 / 19.8 / 12.5 | 0.876 |

Reproduce with:

```bash
python run.py --stages infer measure \
    --training_config conf/tuning/train_st_conformer.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

### Comparison

`egs2/must_c/st1` publishes no results, so there is no like-for-like reference.
The nearest published number is `egs2/must_c_v2/st1`'s `decode_st_conformer`
row -- the same architecture and the same `valid.acc.ave_10best` selection --
at **25.7** BLEU on tst-COMMON, against **24.22** here. That row is MuST-C
**v2**, a larger release than the **v1.2** this recipe uses, so the gap is not
a like-for-like deficit -- treat it as indicative rather than a target.


## Usage

```python
from egs3.must_c.st.dataset import Dataset, DatasetBuilder

builder = DatasetBuilder()
assert builder.is_source_prepared(recipe_dir=".")

# tgt_lang="de" reproduces egs2. Omitting it takes the builder default
# `all`, which aggregates every installed pair -- dev is then 17,740
# examples over 14 pairs instead of en-de's 1,423.
train = Dataset(split="train", recipe_dir=".", tgt_lang="de")   # 229,703
test = Dataset(split="test", recipe_dir=".", tgt_lang="de")     # 2,641, aliased
                                                                # to tst-COMMON
```
