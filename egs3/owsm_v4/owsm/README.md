# owsm_v4 data recipe

This egs3 data module builds **OWSM v4** training data from the raw corpora —
no `egs2/owsm_v4/s2t1/local/` preparation and no published dump is required at
runtime. The egs2 prep scripts are the specification; the published dump is
used only as an offline test oracle.

Two corpora are wired up so far, one directory each under
`dataset/sub_datasets/`:

| corpus | splits | utterances |
|---|---|---|
| SPGISpeech | `train`, `val` | 1,966,109 / 39,341 |
| MuST-C v1.2 (14 language pairs) | `train`, `dev` | 1,106,123 / 6,263 |

## What one utterance looks like

OWSM packs short segments into spans of at most 30 s and conditions each span on
the previous one. A cached row carries the pieces, and `OWSMDataset` composes
the prefix on read:

```
utt_id    MuST-C_v1.2_ted_767_000012750_000041270_en_asr
wav_path  <corpus>/en-ar/data/dev/wav/ted_767.wav
start/end 12.75 .. 41.27
lang/task en / asr          tgt_lang  (empty for ASR, e.g. "de" for translation)
text      <0.00> I'm going to talk today about energy and climate.<3.50><4.28> And ...
text_prev <na>              text_ctc  I'm going to talk today about energy ...
```

**The id and the text spell the language differently, and that is not a bug.**
The id keeps the two-letter code (`_en_asr`), because
`merge_short_utterances` mints it before
`egs2/owsm_v3/s2t1/local/filter_lang_id.py` rewrites the *text* prefix to ISO
639-3 (`<eng>`, `<st_deu>`). Anything deriving both from one field produces ids
that do not match the published data. Porting `owsm_v1/local/utils.py` alone
does not give v4 data.

`text` is stored **without** the `lang`+`task` prefix, so the ISO scheme can
change without rebuilding the cache and `lang`/`task` stay filterable.

## Corpus locations

`path.sh` documents three environment variables; none has a committed default:

| variable | points at |
|---|---|
| `SPGISPEECH` | the root holding `train.csv`, `val.csv` and `spgispeech/{train,val}/` |
| `MUST_C` | the root holding `en-<lang>/data/<split>/txt/` |
| `OWSM_CACHE` | where the built caches go; defaults to `<recipe>/data/hf` |

Point `OWSM_CACHE` at a large shared filesystem — the two corpora together cache
3.07M rows. Neither corpus can be fetched non-interactively: SPGISpeech is
behind a registration form, MuST-C behind a licence form.

```bash
export SPGISPEECH=/path/to/spgispeech
export MUST_C=/path/to/must-c_v1.2
export OWSM_CACHE=/scratch/owsm_cache
. ./path.sh
python run.py --stages create_dataset \
    --training_config conf/tuning/train_owsm_v4_medium.yaml
```

Each corpus is built once and served from
`<OWSM_CACHE>/<corpus>/hf_audio_index/<split>/`, written to a temporary
directory and renamed into place, so an interrupted build leaves no partial
index. Rows whose audio cannot be read are skipped and logged
to `<split>.failures.jsonl` rather than aborting a two-million-row build; all
four splits above built with zero failures.

## Two findings the port depends on

**MuST-C generates the ASR side once per language pair, and upstream throws 77%
of it away.** `collect_data` runs inside `for lang in languages` with a fresh
`wav2utts`, so the `{wav}.asr` group is rebuilt for every pair;
`prepare_must-c.sh` then runs `sort -k1,1 -u` and collapses them by utterance
id. On dev, 14 pairs generate 5,108 ASR utterances that dedup to 1,155, against
5,108 ST utterances — 6,263 rows in all. Skip the dedup and the ASR side is 4.4x
too large and the task mixture the model trains on changes silently.

**The duplicates are not identical, and the winner is arbitrary.** The pairs
segment the same talk independently, so the same 30 s window is packed from
different segment boundaries: the id collides while the text inside differs.
This recipe picks the first language in `sorted(languages)`, which is
deterministic but will not always match the published dump — upstream's winner
falls out of `iterdir()` order crossed with `sort -u` tie-breaking, and `text`,
`text.prev` and `text.ctc` were deduped in separate `sort` invocations, so the
dump can even be internally inconsistent across streams for those rows.

**SPGISpeech durations come from the csv, not from the audio.** Upstream's
`librosa.get_duration(filename=...)` no longer exists in librosa >= 0.10, and
`soundfile.info()` runs at about 15 files/s on a parallel filesystem — 36 hours
for train. The corpus is uniform 16 kHz mono 16-bit PCM, so
`(wav_filesize - 44) / 32000` gives the duration exactly: since the id encodes
`round(1000 * end_time)`, all 2,005,450 published ids reproduce byte for byte.
Every file's size is still checked against the manifest;
`read_headers: true` additionally opens each file to confirm the sample rate.

## Agreement with the published dump

Checked against the full per-corpus dumps with `local/check_against_dump.py`
(not committed — it needs a dump upstream does not ship). Acceptance is
`dump ⊆ ours`: extras are rows that `s2t.sh` stage 4 drops under
`min_wav_duration`, downstream of the prep scripts this module replaces.

| | dump | ours | missing | extra |
|---|---|---|---|---|
| SPGISpeech val | 39,341 | 39,341 | 0 | 0 |
| SPGISpeech train | 1,966,109 | 1,966,109 | 0 | 0 |
| MuST-C dev | 6,248 | 6,263 | 0 | 15 |
| MuST-C train | 1,102,742 | 1,106,123 | 0 | 3,381 |

All three streams are byte-exact on SPGISpeech. On MuST-C every stream mismatch
is a variant we generate — 193/193 on `text`, 129/129 on `text.prev`, 43/43 on
`text.ctc` — which is the arbitrary-winner problem above, not a porting error.

**The extras are not fully explained.** On dev, 11 of 15 are one 50 ms
`(Applause)` span that `min_wav_duration` drops. On train, `min_wav_duration`
accounts for 2,067 of 3,381, leaving **1,314 rows (0.12%) unexplained**, 87% of
them in three directions (`st_fa` 778, `st_ar` 259, `st_vi` 110). `only dump` is
zero in all 15 pairs, so our output is a strict superset rather than a different
segmentation, and no filter in the egs2 scripts accounts for the remainder.
This is a known open item.

## Adding a corpus

A new corpus is a directory under `dataset/sub_datasets/`, nothing more. The
shared layer carries the rest:

- `dataset/utils.py` — the egs2 utterance layer (`merge_short_utterances`,
  `generate_long_utterances`), the ISO tables, and `nlsyms()`.
- `dataset/builder.py` — `OWSMBuilder`: source resolution, split selection,
  parallel file verification, atomic writes, per-file failure logging.
- `dataset/dataset.py` — `OWSMDataset`: load a split, compose the prefix, read
  the audio span.

A sub-dataset overrides `iter_rows()` and `is_valid_source_root()` and sets its
`config.yaml`; its `dataset.py` is about nine lines naming a cache subdirectory.

`nlsyms()` returns the **full** published v4 inventory — 1,681 symbols: `<na>`,
`<nospeech>`, 151 languages, `<asr>`, 25 translation directions and 1,502
timestamps — not just what the wired-up corpora emit. A vocabulary sized to
today's corpora would have to be rebuilt, and every model retrained, the first
time a corpus arrived with a language it had never seen. A test asserts set
equality and ordering against `espnet/owsm_v4_medium_1B`'s `bpe.model`, so the
inventory cannot drift from the published model silently.

## Training

Three settings move together and are the easiest thing to get wrong:

    batch_size x num_device x accumulate_grad_batches = 320
    limit_train_batches = 10,000 x accumulate_grad_batches

320 is egs2's global batch, and 10,000 is its `num_iters_per_epoch` — but that
counts **optimizer** steps where Lightning counts micro-batches, hence the
second line. Change the device count without the other two and the warmup
schedule silently spans a different number of epochs than egs2's.

**Batching is a flat `batch_size` on a plain torch DataLoader**, as
`egs2/owsm_v4/s2t1` uses, not an espnet2 sampler. `shuffle: true` then draws
across the whole mixture. The alternative, `numel` with a `batch_bins` budget,
looks attractive — it caps padded batch cost — but its sampler sorts on
`shape_files[0]`, and `S2TPreprocessor` pads all speech to one 30 s window, so
`feats_shape` is constant and the sort is a no-op. Batches would come out in
cache order, which puts a whole corpus, and within MuST-C a whole language
pair, into the same gradient step. Leading with `text_shape` would make the
sort meaningful, but a sampler fixes its batches once and reuses them every
epoch either way.

Because nothing reads a `*_shape` file at training time, `collect_stats` is
needed only for `global_mvn`'s `feats_stats.npz`.

An epoch is hours long, so `trainer.callbacks` adds a checkpoint every 1,000
steps and `fit.ckpt_path: last` resumes from it; a job that hits its walltime
mid-epoch would otherwise write nothing.

## Tokenizer

One shared BPE vocabulary over the whole mixture, `nbpe=50000` following
`egs2/owsm_v3/s2t1/run.sh`. The 1,681 special symbols are passed as
`user_defined_symbols` so they survive as single pieces:

```
<eng><asr><0.00> the quick brown fox<2.50>
  -> ['▁', '<eng>', '<asr>', '<0.00>', '▁the', '▁quick', '▁brown', '▁fo', 'x', '<2.50>']
```

The text is gathered by streaming the built cache, with
`tokenizer.sample_size` (reservoir-sampled, seeded) to bound what is written as
more corpora join. On the two corpora here the full text is 862 MB over
3,072,231 lines, with an alphabet of 5,409 characters at
`character_coverage: 1.0`; `tokens.txt` holds 50,002 entries once `<sos>`,
`<eos>` and `<sop>` are appended. `vocab_size` has a floor well above
`1,681 + alphabet`, so a small value fails outright.

Run it **after** `create_dataset` and **before** `collect_stats`: the gather
reads the cache, and the preprocessor needs `tokens.txt` before it can turn
`text` into `text_shape`.

## Scoring

`espnet3/systems/owsm/metrics/` reports **CER, WER and TER**, which is what
`egs2/TEMPLATE/s2t1/s2t.sh` reports. There is no BLEU: no owsm recipe computes
one, and s2t.sh scores translation rows as error rates whether or not that is
informative. `TER` here is an error rate over BPE pieces, *not* sacreBLEU's
translation edit rate that `espnet3/systems/esp2_st` reports under the same
name. Scored with jiwer rather than sclite, so expect small differences from
published numbers.

Two details that change the numbers:

- **Tags are removed from both sides**, which s2t.sh asks for and cannot
  achieve: it scores `data/<dset>/text`, still carrying `<eng><asr><0.00>`,
  against a hypothesis that carries none. `CharTokenizer` can strip them, but
  `WordTokenizer` splits on whitespace first and OWSM glues tags to words
  (`fox<2.50>`), so no token ever equals a symbol, and the BPE pass cannot ask
  at all. A perfect transcription therefore scores a large WER under s2t.sh.
  The pattern is built from the recipe's own `nlsyms` rather than from a guess
  at the tag grammar, so a transcript reading `5 < 10 > 3` survives. Set
  `remove_tags: false` to reproduce s2t.sh literally.
- **No text cleaner by default.** `ref_cleaner` and `hyp_cleaner` are separate,
  as s2t.sh keeps them; `egs2/owsm_v3/s2t1/run.sh:18` suggests `whisper_en` for
  both, commented out. Lowercasing and punctuation handling live there, so by
  default case and punctuation count against the model.

## Parallelism

`create_dataset` stays local — its per-file work is millions of `os.stat` calls,
which threads do at about 833/s, and routing tasks that small through Dask costs
more than it saves.

`collect_stats` must not. `parallel.env: local` plans a single shard whatever
`n_workers` says, and on the dev mixture (45,604 utterances) that is **3 h 20 m**
against **7 m 20 s** for `env: slurm` with 64 workers, for byte-identical
output. The cluster block holds dask-jobqueue `SLURMCluster` arguments —
`cores`, `queue`, `walltime` — not Slurm's own `cpus_per_task`, `partition`,
`time`. The committed config keeps `env: local` because the account is
site-specific.

## Results

None yet. This module covers data preparation through `collect_stats`; training
and the model are a separate change.
