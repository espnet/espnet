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

### omniio dump

The `*_omniio.yaml` configs read the data from an
[omniio](https://github.com/wavlab-speech/omniio) dump instead of the
LibriSpeech tree, through the dataset/builder in
[`dataset/omniio/`](dataset/omniio). Their `create_dataset` locates (or
downloads) the corpus as above, then writes each split with omniio's
`Blob.append`, one blob per modality, both keyed by the utterance ID:

```
<omniio_dump_root>/
  audio/librispeech/<split>/{blob_*.bin, metadata.parquet}   # FLAC, as-is
  text/librispeech/<split>/{blob_*.bin, metadata.parquet}    # zstd transcripts
```

`omniio_dump_root` in the configs is `$OMNIIO_DUMP_ROOT`, else `dump/omniio/`
in this directory; `create_dataset`, the dataset entries and the tokenizer all
use it. A blob is finished once it holds every item and carries a
`.dump_complete` marker, and an interrupted dump is completed on the next run.
Once every split is finished the raw corpus is no longer needed, so a dump made
by another user on the same file system can be used as-is. omniio records
absolute blob paths, so a dump copied elsewhere needs the `path` column of its
`metadata.parquet` files rewritten.

Each dataset entry reads one split through `default_query` in
[`dataset/omniio/config.yaml`](dataset/omniio/config.yaml), a DuckDB query over
the dump's `metadata.parquet` files. An entry may give its own `query` to
filter, join or combine data when the job starts; paths are relative to the
dump root and `$split` is the entry's `split`. For example, to keep only
utterances of 0.5-20 s:

```yaml
- data_src: egs3.librispeech_100.esp2_asr.dataset.omniio
  data_src_args:
    split: train-clean-100
    query: |
      SELECT a.id,
             a.path AS speech_path, a.start_byte AS speech_start,
             a.end_byte - a.start_byte AS speech_size, a.samples AS length,
             t.path AS text_path, t.start_byte AS text_start,
             t.end_byte - t.start_byte AS text_size
      FROM read_parquet('audio/librispeech/' || $split || '/metadata.parquet') AS a
      JOIN read_parquet('text/librispeech/' || $split || '/metadata.parquet') AS t
        USING (id)
      WHERE a.samples / a.sample_rate BETWEEN 0.5 AND 20
      ORDER BY a.id
```

Batches are planned from the query's `length` column (omniio's sample count)
with `batches.lengths_from_dataset: true`, so no shape files are needed;
`collect_stats` still runs for the global mean/variance statistics. The
training rows are split into `total_shards` shards by `hash(id)` and each GPU
loads only its own shard. `dist_world_size` must equal the total number of
GPUs and `total_shards` must be a multiple of it. As with any iterator whose
ranks get different batch counts, every rank ends the epoch at the shortest
rank's count, which drops about 3% of the batches with 8 shards by `id`. To
train on more GPUs, set `num_device` (the shard counts follow it) and divide
`accumulate_grad_batches` or `batch_bins` to keep the effective batch size.

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

To read the data from the omniio dump, use the omniio configs listed under
Results and, optionally, set where the dump lives first:

```bash
export OMNIIO_DUMP_ROOT=/path/to/omniio_dump   # default: dump/omniio
export LIBRISPEECH=/path/to/LibriSpeech        # only if the dump is not built yet
```

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

### E-Branchformer, data from omniio

- Training config: `conf/tuning/training_e_branchformer_omniio.yaml`
- Inference config: `conf/inference_omniio.yaml`
- Metrics config: `conf/metrics.yaml`

The same model, tokenizer and decoding as the E-Branchformer above, with the
data read from the omniio dump (see [omniio dump](#omniio-dump)).

| dataset | WER | CER |
| --- | --- | --- |
| dev-clean | 6.10 | 2.39 |
| dev-other | 16.75 | 8.30 |
| test-clean | 6.45 | 2.45 |
| test-other | 17.01 | 8.28 |

Trained for all 200 configured epochs on 8 NVIDIA H200 GPUs (one node) in
bf16, at about 50 s per epoch (38.49 M parameters). For 8 GPUs the config was
run with `num_device: 8`, `batch_bins: 3800000` and
`accumulate_grad_batches: 1`, the same effective batch size as the 1-GPU
default (see the note at the top of the config). Each GPU loaded one of 8
shards and kept 181 batches per epoch, dropping 1.6-5.2% of its shard's
batches. Decoded with `conf/inference_omniio.yaml` from the
`valid.acc.ave_10best` average of epochs 179-199 (0-based). The results match
the published E-Branchformer results above within run-to-run variation.

#### Pretrained model

No pretrained model has been published for this configuration.
