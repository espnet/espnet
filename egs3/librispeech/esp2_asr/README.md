# LibriSpeech 960h ASR recipe

Port of [`egs2/librispeech/asr1`](../../../egs2/librispeech/asr1) to ESPnet3:
E-Branchformer (17 blocks, d512) on the full 960 h training set, with
[`conf/tuning/train_e_branchformer.yaml`](conf/tuning/train_e_branchformer.yaml)
following
`egs2/librispeech/asr1/conf/tuning/train_asr_e_branchformer.yaml`.

Differences from the egs2 recipe are marked `[DEVIATION]` in that config. The
ones that change the result:

- **No language model.** ESPnet3 has no LM stage, so decoding is
  CTC/attention only and the WER is not comparable to egs2's published number.
- **Speed perturbation on the fly** rather than tripling the dumped data.
- **`max_epochs: 240`, not egs2's 80.** These are the same amount of training.
  `--speed_perturb_factors` dumps the three perturbed copies as separate
  utterances, so an egs2 epoch is three passes over the corpus and an epoch
  here is one. `warmup_steps` is left at egs2's 40000: it counts optimizer
  steps and does not follow the epoch budget.
- **`batch_bins` restated in ESPnet3's per-GPU unit**, which also bins on
  `feats_shape` rather than raw samples.

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

## How the audio is stored

LibriSpeech ships one `.flac` per utterance: 281k files for the 960 h training
set. On the shared filesystems this recipe targets, that file count — not the
60 GB — is what makes training IO-bound, and it exhausts inodes when several
corpora are staged side by side. So the recipe packs the corpus into
[Omni-IO](https://pypi.org/project/omniio/) archives, which ESPnet already
depends on, rather than reading the corpus tree directly.

`create_dataset` packs each **raw** LibriSpeech directory once, and then writes
one small table per logical split:

```
data/archive/raw/train-clean-100/   blob_0.bin, ...  source files verbatim
                                    metadata.parquet id, bin_index, byte range
                                    text.tsv         utt_id <TAB> transcript
data/archive/train_960.parquet      the view: id, source, bin_index, byte range
data/archive/dev.parquet            ... and so on, one per split
```

A split is therefore a view, not a copy. `dev` points at the dev-clean and
dev-other archives, `train_100` and `train_960` share train-clean-100, and
nothing is stored twice however many splits are built.

Reading an utterance is one range read (`omniio.interface.audio_read`), and the
bytes are the original FLAC, so the decoded samples are identical to reading
the corpus directly. The archives are relocatable: the dataset rebuilds blob
paths from `bin_index` rather than trusting the absolute `path` column that
Omni-IO records at packing time.

## Variants

Four training configs, each inheriting the baseline and stating only what it
changes, so any pair isolates one thing:

| config | changes | what it isolates |
| --- | --- | --- |
| `train_e_branchformer.yaml` | — | the baseline, `rel_selfattn` over `rel_pos` |
| `train_e_branchformer_sdpa.yaml` | `use_sdpa` | the attention kernel, same model |
| `train_e_branchformer_abs_pos.yaml` | `selfattn` + `abs_pos` | dropping relative positions |
| `train_e_branchformer_abs_pos_bf16.yaml` | `bf16-true` | fp32 master weights |

## Results

**No WER yet.** The `infer` and `measure` stages have not been run, so there is
nothing to compare against egs2 — and the comparison would not be fair anyway,
since this recipe decodes without a language model.

Training is in progress. What follows is `valid/acc` and wall clock from runs
on **one NVIDIA A40** (`batch_bins: 4375000`, `accumulate_grad_batches: 16`,
391 optimizer steps per epoch), as of 2026-10-14:

| config | min/epoch | ep 9 | ep 17 | ep 25 | ep 34 | ep 45 |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | 106 | 0.436 | 0.901 | 0.928 | 0.939 | — |
| `_sdpa` | 96 | 0.466 | 0.899 | 0.926 | — | — |
| `_abs_pos` | 80 | 0.397 | 0.884 | 0.921 | 0.935 | 0.945 |
| `_abs_pos_bf16` | 70 | 0.120 | 0.424 | 0.661 | 0.800 | — |

Four things these say so far:

- **SDPA does not change the learning.** The baseline and `_sdpa` track each
  other for 28 epochs — 0.920 against 0.920 at epoch 21, 0.934 against 0.934 at
  28. Only the kernel differs, and it is 10% faster per epoch.
- **Relative positions buy a little accuracy per epoch, and the gap closes.**
  `_abs_pos` trails the baseline by 0.017 at epoch 17, 0.007 at 25, 0.004 at
  34.
- **Per hour, `_abs_pos` and `_sdpa` come out level.** At ~45 h `_sdpa` is at
  0.934 and `_abs_pos` is at 0.934; at 60 h the baseline is at 0.939 and
  `_abs_pos` at 0.945. Making relative attention fast is worth as much as
  giving it up.
- **`bf16-true` is slow to start but does converge.** It sits on the opening
  plateau until epoch 14 and lifts around 17, where the others lift at 9; by
  epoch 37 it is at 0.838 and still climbing, roughly 0.10 behind the others at
  matched epochs and closing. Dropping the fp32 master weights costs the early
  updates, when warmup keeps them smallest and bf16's 7-bit mantissa drops
  them. Whether 240 epochs is enough for it to catch up is still open.

## Pretrained Models

None.
