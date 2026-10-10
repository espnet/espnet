# LibriSpeech 960h ASR recipe

Port of [`egs2/librispeech/asr1`](../../../egs2/librispeech/asr1) to ESPnet3:
E-Branchformer (17 blocks, d512) on the full 960 h training set, with
[`conf/tuning/train_e_branchformer.yaml`](conf/tuning/train_e_branchformer.yaml)
following
`egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml`.

Differences from the egs2 recipe are marked `[DEVIATION]` in that config. The
three that change the result: **no language model** (ESPnet3 has no LM stage,
so decoding is CTC/attention only), speed perturbation applied **on the fly**
instead of tripling the dumped data, and `batch_bins` restated in ESPnet3's
per-GPU unit.

## Requirements

This recipe stores audio in [Omni-IO](https://pypi.org/project/omniio/)
archives, which is not among ESPnet's default dependencies:

```bash
pip install omniio
```

## Corpus

Point `LIBRISPEECH` at an existing LibriSpeech root — the directory that holds
`train-clean-100/`, `dev-clean/`, ... — before the first stage:

```bash
export LIBRISPEECH=/path/to/LibriSpeech
```

Without it the recipe looks under `download/LibriSpeech`, and
`create_dataset.download: true` fetches the ~60 GB of OpenSLR archives there.

`data/` and `exp/` take the packed archives (roughly the size of the corpus),
the statistics and the checkpoints. On a cluster, symlink them to fast storage
before running anything, rather than editing paths in the config:

```bash
ln -s /path/to/fast/storage/librispeech/data data
ln -s /path/to/fast/storage/librispeech/exp  exp
```

## Quick start

```bash
# 1) Pack the corpus into Omni-IO archives (data/archive/<split>/)
python run.py --stages create_dataset \
    --training_config conf/tuning/train_e_branchformer.yaml

# 2) Train the BPE-5000 tokenizer on the packed train_960 transcripts
python run.py --stages train_tokenizer \
    --training_config conf/tuning/train_e_branchformer.yaml

# 3) Collect feature statistics (global_mvn + batch shapes)
python run.py --stages collect_stats \
    --training_config conf/tuning/train_e_branchformer.yaml

# 4) Train
python run.py --stages train \
    --training_config conf/tuning/train_e_branchformer.yaml

# 5) Decode and score
python run.py --stages infer measure \
    --training_config conf/tuning/train_e_branchformer.yaml \
    --inference_config conf/inference.yaml \
    --metrics_config conf/metrics.yaml
```

`collect_stats` over 281k utterances is slow with a single local worker: set
`parallel.env` to a cluster backend and supply your scheduler's `options`
(queue, account, walltime) in the training config.

## Splits

`dataset/config.yaml` defines the logical splits, named after
`egs2/librispeech/asr1/run.sh`:

| split | raw directories |
| --- | --- |
| `train_960` | train-clean-100, train-clean-360, train-other-500 |
| `train_460` | train-clean-100, train-clean-360 |
| `train_100` | train-clean-100 |
| `dev` | dev-clean, dev-other |
| `dev_clean`, `dev_other`, `test_clean`, `test_other` | the matching directory |

`create_dataset` builds only the splits listed in the training config, so a
100 h debug run does not need the 360 h and 500 h archives staged.

## Why Omni-IO

LibriSpeech ships one `.flac` per utterance: 281k files for the 960 h training
set. On the shared filesystems this recipe targets, that file count — not the
60 GB — is what makes training IO-bound, and it exhausts inodes when several
corpora are staged side by side. `create_dataset` packs each split into
`data/archive/<split>/`:

```
blob_0.bin, blob_1.bin, ...   source files stored verbatim, 320 MiB per blob
metadata.parquet              id, bin_index, start_byte, end_byte per utterance
text.tsv                      utt_id <TAB> transcript
```

Reading an utterance is one range read (`omniio.interface.audio_read`), and the
bytes are the original FLAC, so the decoded samples are identical to reading
the corpus directly. The archive is relocatable: the dataset rebuilds blob
paths from `bin_index` rather than trusting the absolute `path` column that
Omni-IO records at packing time.

## Results

Not run yet. This port has produced no numbers, so none are quoted here; the
egs2 recipe's published WER is not comparable anyway, because this recipe
decodes without a language model.

## Pretrained Models

None.
