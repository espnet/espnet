# LibriSpeech 100h ASR recipe

## Data Preparation

`create_dataset` downloads the five OpenSLR archives (`train-clean-100`,
`dev-clean`, `dev-other`, `test-clean`, `test-other`) into `download/` and
extracts them to `download/LibriSpeech`:

```bash
python run.py --stages create_dataset \
    --training_config conf/tuning/training_e_branchformer.yaml
```

To reuse a corpus you already have, point the recipe at it before running
`create_dataset`; either the `LibriSpeech` directory or its parent works.
Locations are checked in this order and the first that holds a corpus wins:

1. `builder.dataset_path` in `dataset/config.yaml` (default `download`,
   relative to this directory)
2. `source_dir` passed to the builder or dataset
3. the `LIBRISPEECH` environment variable

When all splits are present nothing is downloaded. Missing splits are
downloaded only into `builder.dataset_path`, and an existing archive there is
extracted instead of downloaded again. A corpus found through `source_dir` or
`LIBRISPEECH` is never modified; if it is missing splits, `create_dataset`
reports them.

## Commands

Run the following commands from `egs3/librispeech_100/esp2_asr/`. `path.sh`
activates the tools environment and puts this checkout on `PYTHONPATH`. Set
the two config variables to the ones listed under [Results](#results) for the
model you want; the defaults train the E-Branchformer.

```bash
. ./path.sh
# Set these to the configs listed under Results for the model you want.
train_config=conf/tuning/training_e_branchformer.yaml
inference_config=conf/inference.yaml

python run.py --stages create_dataset train_tokenizer --training_config $train_config
python run.py --stages collect_stats --training_config $train_config
python run.py --stages train --training_config $train_config
python run.py --stages infer measure --training_config $train_config \
    --inference_config $inference_config --metrics_config conf/metrics.yaml
```

Run `collect_stats` and `train` as separate commands: `collect_stats` removes
`model.normalize` from the training config it was given, so a `train` stage
in the same command builds the model without global mean/variance
normalization.

Both models use the same data, BPE-5000 tokenizer and frontend, so `data/`
and `exp/stats` are shared: once one model has run `collect_stats`, the other
can start from `train`.

## Results

### E-Branchformer

- Training config: `conf/tuning/training_e_branchformer.yaml`
- Inference config: `conf/inference.yaml`
- Metrics config: `conf/metrics.yaml`

Checkpoints are ranked by `valid/acc` and the 10 best are averaged into
`valid.acc.ave_10best.pth`. `conf/inference.yaml` decodes that average with
beam 20 and `ctc_weight` 0.3.

The [published model card](https://huggingface.co/ms180/librispeech_100h_e_branchformer)
reports the following results for an E-Branchformer with 38.47 M parameters.
They were produced with an earlier version of this recipe; rerunning the
current config has not been verified to reproduce them.

| dataset | WER | CER |
| --- | --- | --- |
| dev-clean | 6.36 | 2.55 |
| dev-other | 16.80 | 8.33 |
| test-clean | 6.41 | 2.49 |
| test-other | 17.12 | 8.31 |

#### Pretrained model

- [`ms180/librispeech_100h_e_branchformer`](https://huggingface.co/ms180/librispeech_100h_e_branchformer)

### Conformer-CTC

- Training config: `conf/tuning/training_conformer_ctc.yaml`
- Inference config: `conf/inference_ctc.yaml`
- Metrics config: `conf/metrics.yaml`

`conf/tuning/training_conformer_ctc.yaml` ports
`egs2/librispeech_100/asr1/conf/tuning/train_conformer_ctc.yaml`: an 18-block
Conformer encoder trained with the CTC loss only (no attention decoder).
Without a decoder `valid/acc` is not logged, so checkpoints are ranked by
`valid/cer_ctc` and the 10 best are averaged into
`exp/training_conformer_ctc/valid.cer_ctc.ave_10best.pth`.
`conf/inference_ctc.yaml` decodes with the CTC prefix scorer alone
(`ctc_weight: 1.0`, `beam_size: 1`, as egs2's `decode_ctc_bs1.yaml`) and
writes to `exp/training_conformer_ctc/inference_ctc/<test set>/`.

| dataset | WER | CER |
| --- | --- | --- |
| dev-clean | 10.45 | 4.43 |
| dev-other | 23.95 | 12.49 |
| test-clean | 10.72 | 4.53 |
| test-other | 25.18 | 12.94 |

Trained for all 180 configured epochs on 1 NVIDIA A100 (40 GB) in bf16, at
about 3.9 min per epoch (31.7 M parameters). Training was interrupted once
after epoch 30 and resumed from the step checkpoint with `fit.ckpt_path=last`;
the step-based `WarmupLR` schedule continues unchanged across the restart.
Decoded with `conf/inference_ctc.yaml` (CTC only, `beam_size` 1, no LM) from
the `valid.cer_ctc.ave_10best` average of epochs 156-178 (0-based, as in the
checkpoint file names).

egs2 reports no result for `train_conformer_ctc.yaml`. Its closest CTC-only
baseline, `train_asr_ctc_e_branchformer_e12_mlp1024_linear1024.yaml` (26.4 M
parameters, `decode_ctc_bs1`), reaches WER 9.2 / 22.4 / 9.6 / 23.1 on the same
four sets.

#### Pretrained model

- [`YuCeong-May/Conformer-CTC`](https://huggingface.co/YuCeong-May/Conformer-CTC)

#### Differences from the egs2 recipe

These are deliberate; each is marked `[DEVIATION]` in the training config.

- **Frontend window shared with the E-Branchformer config.** `frontend_conf`
  uses `n_fft` 512 / `win_length` 400 / `hop_length` 160 instead of the
  espnet2 default (`win_length` 512, `hop_length` 128), so `exp/stats` is
  shared with `training_e_branchformer.yaml`.
- **`batch_bins` is not egs2's `2000000`.** egs2 bins on raw samples with
  `accum_grad: 16`; ESPnet3 bins on `feats_shape` (frames x 80), so this
  config uses `batch_bins: 4000000` (about 500 s of audio) with
  `accumulate_grad_batches: 4`, keeping about 2,000 s of audio per update.
- **Speed perturbation on the fly, 180 epochs instead of 60.** The
  0.9/1.0/1.1 perturbation is applied by the preprocessor instead of tripling
  the training set on disk. An epoch therefore covers 100 h instead of 300 h,
  so `max_epochs` is 180 to keep the same number of updates.

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
