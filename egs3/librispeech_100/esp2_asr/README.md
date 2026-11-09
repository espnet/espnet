# LibriSpeech 100h ASR recipe

Place the corpus under `download/LibriSpeech`, or set the `LIBRISPEECH`
environment variable to an existing LibriSpeech root, before running
`create_dataset`/`train`.

## Quick start

```bash
# 1) Train the default E-Branchformer model
python run.py --stages train \
    --training_config conf/tuning/training_e_branchformer.yaml

# 2) Decode
python run.py --stages infer \
    --training_config conf/tuning/training_e_branchformer.yaml \
    --inference_config conf/inference.yaml

# 3) Score
python run.py --stages measure \
    --training_config conf/tuning/training_e_branchformer.yaml \
    --metrics_config conf/metrics.yaml
```

## Reading data from omniio

[`conf/tuning/training_e_branchformer_omniio.yaml`](conf/tuning/training_e_branchformer_omniio.yaml)
trains the same model with its data read from an
[omniio](https://github.com/wavlab-speech/omniio) dump instead of the raw
LibriSpeech tree. It only switches `data_src` to the dataset/builder in
[`dataset/omniio/`](dataset/omniio), the tokenizer text builder and batching;
everything else is inherited from `training_e_branchformer.yaml`.

```bash
. ./path.sh
export CORPORA_ROOT=/path/to/corpora          # optional
export OMNIIO_DUMP_ROOT=/path/to/omniio_dump  # optional

python run.py --stages create_dataset train_tokenizer collect_stats train \
    --training_config conf/tuning/training_e_branchformer_omniio.yaml
python run.py --stages infer \
    --training_config conf/tuning/training_e_branchformer_omniio.yaml \
    --inference_config conf/inference_omniio.yaml
```

### Data layout

`create_dataset` does two things:

1. **Raw corpus.** It looks for `<corpora_root>/librispeech/LibriSpeech/<split>`
   and downloads only the splits that are missing from OpenSLR, checking each
   archive's md5. `<corpora_root>` is `create_dataset.corpora_root`, else
   `$CORPORA_ROOT`, else `download/` in this recipe.
2. **omniio dump.** Each split is written with omniio's `Blob.append`, one blob
   per modality:

   ```
   <dump_root>/
     audio/librispeech/<split>/{blob_*.bin, metadata.parquet}   # FLAC, as-is
     text/librispeech/<split>/{blob_*.bin, metadata.parquet}    # zstd transcripts
   ```

   Both blobs use the LibriSpeech utterance ID (`103-1240-0000`) as `id`, so
   an utterance and its transcript are the rows with the same ID. Other
   corpora and modalities go next to them as `<modality>/<corpus>/...`.
   `<dump_root>` is `create_dataset.dump_root`, else `$OMNIIO_DUMP_ROOT`,
   else `dump/omniio/` in this recipe.

Both steps skip work already done, so a re-run, or a run against a dump made
by someone else, does nothing.

### Selecting data with SQL

Training, tokenizer training and inference read only the dump, through
`espnet3.components.data.omniio_dataset.OmniIODataset`. Each dataset entry in
the config is a DuckDB query over the dump's `metadata.parquet` files, so data
is selected, joined and filtered when the job starts, not when it is dumped:

```yaml
dataset:
  train:
    - data_src: egs3.librispeech_100.esp2_asr.dataset.omniio
      data_src_args:
        query: |
          SELECT a.id,
                 a.path AS speech_path, a.start_byte AS speech_start,
                 a.end_byte - a.start_byte AS speech_size,
                 (a.samples // 160 + 1) * 80 AS length,
                 t.path AS text_path, t.start_byte AS text_start,
                 t.end_byte - t.start_byte AS text_size
          FROM read_parquet('{dump_root}/audio/librispeech/train-clean-100/metadata.parquet') AS a
          JOIN read_parquet('{dump_root}/text/librispeech/train-clean-100/metadata.parquet') AS t
            USING (id)
          WHERE a.samples / a.sample_rate BETWEEN 0.5 AND 20   -- optional filter
          ORDER BY a.id
```

The query returns one row per item:

| Column | Meaning |
|---|---|
| `id` | item ID |
| `<field>_path`, `<field>_start`, `<field>_size` | byte range of `speech` / `text` in an omniio blob, decoded with `omniio.interface` |
| `<field>` (instead of the three above) | an inline value used as-is, e.g. text from a parquet table |
| `length` | size used for batching (`batches.lengths_from_dataset: true`) |

`{name}` placeholders are filled from the entry's `params`; `{dump_root}` is
filled automatically. Entries with only `split: ...` (validation and
test) use `default_query` from [dataset/omniio/config.yaml](dataset/omniio/config.yaml).
Adding a split is a `UNION ALL`, and any metadata column (`samples`,
`sample_rate`, `format`, ...) can filter rows.

#### One index shard per GPU

The training rows are split into `total_shards` shards by `hash(id)` (set
`shard_by` to keep e.g. a speaker's rows together). Each GPU runs the query for
its own shard only, so no rank holds the full index, and batches are planned
from that shard's `length` column instead of collect_stats shape files. Ranks
keep the same number of batches each epoch, dropping a few at random when
shards differ in size. `dataloader.train.dist_world_size` must equal the
total number of GPUs (`num_device * num_nodes`), and `total_shards` must be a
positive multiple of it; with more shards than GPUs, ranks rotate through the
shards from epoch to epoch.

## Pretrained Models

- [`conf/tuning/training_e_branchformer.yaml`](https://huggingface.co/ms180/librispeech_100h_e_branchformer)
