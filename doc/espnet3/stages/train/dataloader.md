---
title: ESPnet3 Dataloader and Collate
author:
  name: "Masao Someki"
date: 2025-11-26
---

# ESPnet3 Dataloader and Collate

This page summarizes how `dataloader` and `collate_fn` work in ESPnet3. It
supports both the ESPnet iterator setup and the standard PyTorch DataLoader; we
explain the ESPnet flow first. For
full configuration options, see
[train config reference](../../config/train_config.md).

In training, these dataloaders are built inside the LightningModule
implementation: `espnet3/components/modeling/lightning_module.py`
(`ESPnetLightningModule`).

## Dataloader config overview (ESPnet iterator)

Start with the dataloader config block. This is the default ESPnet iterator
setup used by `collect_stats` and `train` stage:

```yaml
dataloader:
  collate_fn:
    _target_: espnet2.train.collate_fn.CommonCollateFn
    int_pad_value: -1
  train:
    multiple_iterator: false
    num_shards: 1
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: true
      collate_fn: ${dataloader.collate_fn}
      batches:
        type: sorted
        shape_files:
          - ${stats_dir}/train/feats_shape
  valid:
    multiple_iterator: false
    num_shards: 1
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: false
      collate_fn: ${dataloader.collate_fn}
      batches:
        type: ${dataloader.train.iter_factory.batches.type}
        shape_files:
          - ${stats_dir}/valid/feats_shape
```

## Collate function contract

A collate function takes a list of dataset samples and turns them into a single
batch (padding, stacking, and length bookkeeping). In ESPnet3 you can provide a
custom collate function, or use ESPnet2's `CommonCollateFn`.

For `CommonCollateFn`, see the
[CommonCollateFn implementation](https://espnet.github.io/espnet/guide/espnet2/train/CommonCollateFn.html).

Example (what it does):

```python
from espnet2.train.collate_fn import CommonCollateFn

collate = CommonCollateFn(int_pad_value=-1)
items = [
    ("utt1", {"speech": np.ones((3,)), "text": np.array([1, 2, 3])}),
    ("utt2", {"speech": np.ones((5,)), "text": np.array([4, 5])}),
]
uids, batch = collate(items)

# uids == ["utt1", "utt2"]
# batch["speech"].shape == (2, 5)
# batch["speech_lengths"] == [3, 5]
# batch["text"].shape == (2, 3)
# batch["text_lengths"] == [3, 2]
```

`batch` is a dictionary that contains batched arrays such as `speech` and
`text`, plus the matching length fields (e.g., `speech_lengths`,
`text_lengths`) computed from the original samples.
The arrays are padded to the max length in the batch, and the `*_lengths`
fields preserve the original lengths (used for attention masks, etc.).

These keys are passed directly into model methods like `forward()` and
`collect_feats()` during training and stats collection.

## Custom collate function

If you need custom batching logic, you can implement your own collate function.
The expected input/output is:

- **Input**: list of `(uid, sample_dict)` items.
- **Output**: `(uids, batch)` where `batch` is a dict of tensors/arrays.

Example: add white noise to `speech` before calling `CommonCollateFn`:

```python
from espnet2.train.collate_fn import CommonCollateFn

class MyCustomCollateFn:
    def __init__(self, int_pad_value=-1, noise_std=0.005):
        self.base = CommonCollateFn(int_pad_value=int_pad_value)
        self.noise_std = noise_std

    def __call__(self, items):
        noisy_items = []
        for uid, sample in items:
            sample = dict(sample)
            speech = sample["speech"]
            noise = np.random.normal(0.0, self.noise_std, size=speech.shape)
            sample["speech"] = speech + noise
            noisy_items.append((uid, sample))
        return self.base(noisy_items)
```

If the collate function is recipe-specific, define it under `egs3/<recipe>/<task>/src/`
and reference it in `train.yaml`:

```yaml
dataloader:
  collate_fn:
    _target_: src.my_collate.MyCustomCollateFn
```

## Standard PyTorch DataLoader

If you prefer to use the standard PyTorch DataLoader, disable the ESPnet
iterator settings by setting the iterator-related fields to `null`, and then
provide the usual DataLoader arguments:

```yaml
dataloader:
  collate_fn:
    _target_: espnet2.train.collate_fn.CommonCollateFn
    int_pad_value: -1
  train:
    iter_factory: null
    batch_size: 8
    num_workers: 4
    shuffle: true
  valid:
    iter_factory: null
    batch_size: ${dataloader.train.batch_size}
    num_workers: ${dataloader.train.num_workers}
    shuffle: false
```

## Iterator + batches settings (ESPnet)

For efficient batching based on `collect_stats`, we recommend using the
ESPnet2 iterator/sampler implementations. See:

- `espnet2/iterators/`
- `espnet2/samplers/`

The `iter_factory` section controls how batches are created. The `batches`
subsection decides how to group samples, often using the shape files produced
by `collect_stats`:

```yaml
dataloader:
  train:
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: true
      collate_fn: ${dataloader.collate_fn}
      batches:
        type: sorted
        shape_files:
          - ${stats_dir}/train/feats_shape
        batch_size: 16
        batch_bins: 12000000
```

### Iterator factories (ESPnet2)

| Iterator | Supported batch types | Description |
| --- | --- | --- |
| [`SequenceIterFactory`](https://espnet.github.io/espnet/guide/espnet2/iterators/SequenceIterFactory.html) | `unsorted`, `sorted`, `folded`, `length`, `numel` | Standard iterator that builds DataLoader batches from precomputed `batches` and keeps shuffling reproducible across epochs. |
| [`ChunkIterFactory`](https://espnet.github.io/espnet/guide/espnet2/iterators/ChunkIterFactory.html) | Per-sample batches (`batch_size: 1`) | Splits long sequences into chunks for training with fixed-length windows and overlap. |
| [`CategoryIterFactory`](https://espnet.github.io/espnet/guide/espnet2/iterators/CategoryIterFactory.html) | `catbel`, `catpow`, `catpow_balance_dataset` | Balances batches across categories/classes using category-aware samplers to reduce skew. |
| [`CategoryChunkIterFactory`](https://espnet.github.io/espnet/guide/espnet2/iterators/CategoryChunkIterFactory.html) | Per-sample batches (`batch_size: 1`) | Combines category balancing with chunked iteration for long-sequence tasks. |
<!-- | [`MultipleIterFactory`](https://espnet.github.io/espnet/guide/espnet2/iterators/MultipleIterFactory.html) | Depends on wrapped iterators | Chains multiple iterators to mix different datasets or sampling strategies in one epoch. | -->

#### SequenceIterFactory

Use this for standard sequence batching. It works with the common `batches`
types like `sorted`, `unsorted`, `folded`, `length`, and `numel`.

```yaml
dataloader:
  train:
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: true
      collate_fn: ${dataloader.collate_fn}
      batches:
        type: sorted
        shape_files:
          - ${stats_dir}/train/feats_shape
```

#### ChunkIterFactory

Use this when you want fixed-length chunks from long sequences. It builds
chunks before collation.

```yaml
dataloader:
  train:
    iter_factory:
      _target_: espnet2.iterators.chunk_iter_factory.ChunkIterFactory
      batch_size: 16
      chunk_length: 800
      batches:
        - [utt1]
        - [utt2]
```

#### CategoryIterFactory

Use this when you need category-balanced sampling. It pairs with `catbel`,
`catpow`, or `catpow_balance_dataset`.

```yaml
dataloader:
  train:
    iter_factory:
      _target_: espnet2.iterators.category_iter_factory.CategoryIterFactory
      batch_type: catbel
      sampler_args:
        category2utt_file: ${stats_dir}/train/utt2category
        batch_size: 32
```

#### CategoryChunkIterFactory

Use this for category-balanced chunking (long sequences + category balancing).

```yaml
dataloader:
  train:
    iter_factory:
      _target_: espnet2.iterators.category_chunk_iter_factory.CategoryChunkIterFactory
      batch_size: 8
      chunk_length: 800
      batch_type: catbel
      sampler_args:
        category2utt_file: ${stats_dir}/train/utt2category
        batch_size: 32
```

## Sharded iteration (multiple_iterator)

Sharding means splitting a huge dataset into smaller pieces (shards) so you
don't have to load or iterate the entire dataset at once. This becomes
important at trainin with million‑hour scale data where loading/training with
the entire data every epoch is too heavy.

When `multiple_iterator: true`, ESPnet3 selects one shard per epoch and builds
the iterator on that shard only. `num_shards` controls how many pieces you
split the dataset into:

- `num_shards: 1` keeps the full dataset as a single shard (no sharding).
- `num_shards: 10` splits the dataset into 10 parts and uses one part per epoch.

```yaml
dataloader:
  train:
    multiple_iterator: true
    num_shards: 10
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: true
      collate_fn: ${dataloader.collate_fn}
      batches:
        type: sorted
        shape_files:
          - ${stats_dir}/train/feats_shape.{shard_idx}
  valid:
    multiple_iterator: true
    num_shards: 10
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: false
      collate_fn: ${dataloader.collate_fn}
      batches:
        type: sorted
        shape_files:
          - ${stats_dir}/valid/feats_shape.{shard_idx}
```

### Batch samplers (ESPnet2)

The `batches` config maps to ESPnet2 samplers that build batch indices from
shape files.

| Sampler | Description |
| --- | --- |
| [`SortedBatchSampler`](https://espnet.github.io/espnet/guide/espnet2/samplers/SortedBatchSampler.html) | Sorts by length and groups similar-length samples to reduce padding. |
| [`UnsortedBatchSampler`](https://espnet.github.io/espnet/guide/espnet2/samplers/UnsortedBatchSampler.html) | Creates batches without sorting (simple/random order). |
| [`FoldedBatchSampler`](https://espnet.github.io/espnet/guide/espnet2/samplers/FoldedBatchSampler.html) | Forms batches by folding sorted lists to keep length variation balanced. |
| [`LengthBatchSampler`](https://espnet.github.io/espnet/guide/espnet2/samplers/LengthBatchSampler.html) | Batches by length constraints (e.g., max frames). |
| [`NumElementsBatchSampler`](https://espnet.github.io/espnet/guide/espnet2/samplers/NumElementsBatchSampler.html) | Batches by total elements (e.g., frame count) instead of fixed batch size. |
| [`CategoryBalancedSampler`](https://espnet.github.io/espnet/guide/espnet2/samplers/CategoryBalancedSampler.html) | Balances categories/classes per batch. |
| [`CategoryPowerSampler`](https://espnet.github.io/espnet/guide/espnet2/samplers/CategoryPowerSampler.html) | Category sampling with power-law smoothing. |
| [`CategoryDatasetPowerSampler`](https://espnet.github.io/espnet/guide/espnet2/samplers/CategoryDatasetPowerSampler.html) | Dataset-level power sampling combined with category sampling. |

### Sampler config examples

#### SortedBatchSampler

```yaml
dataloader:
  train:
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      batches:
        type: sorted
        shape_files:
          - ${stats_dir}/train/feats_shape
```

#### UnsortedBatchSampler

```yaml
dataloader:
  train:
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      batches:
        type: unsorted
        shape_files:
          - ${stats_dir}/train/feats_shape
```

#### FoldedBatchSampler

`fold_lengths` tells the sampler what length thresholds to use when shrinking
batch size for long sequences. `batch_size` is the base size for short samples,
and `min_batch_size` prevents the batch size from becoming too small when
sequences are very long.

For example, if `batch_size: 32`, `min_batch_size: 1`, and `fold_lengths: [800]`,
then a batch with max length around 800 keeps size 32, while much longer
sequences will reduce the batch size (but never below 1).

```yaml
dataloader:
  train:
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      batches:
        type: folded
        shape_files:
          - ${stats_dir}/train/feats_shape
        batch_size: 32
        min_batch_size: 1
        fold_lengths:
          - 800
```

#### LengthBatchSampler

`batch_bins` sets the target total length per batch. The sampler groups samples
so the sum of lengths in a batch stays near this value.

```yaml
dataloader:
  train:
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      batches:
        type: length
        shape_files:
          - ${stats_dir}/train/feats_shape
        batch_bins: 12000000
```

#### NumElementsBatchSampler

`batch_bins` sets the target total element count per batch (e.g., frames × dims),
so batches have similar overall size even if sequence lengths differ.

```yaml
dataloader:
  train:
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      batches:
        type: numel
        shape_files:
          - ${stats_dir}/train/feats_shape
        batch_bins: 12000000
```

#### CategoryBalancedSampler

CategoryBalancedSampler keeps class/category balance within each batch. Use it
when you want each minibatch to contain a more even mix of categories.

```yaml
dataloader:
  train:
    iter_factory:
      _target_: espnet2.iterators.category_iter_factory.CategoryIterFactory
      batch_type: catbel
      sampler_args:
        category2utt_file: ${stats_dir}/train/utt2category
        batch_size: 32
        min_batch_size: 1
```

`utt2category` is a simple mapping from category to utterance IDs, for example:

```
cat_a utt1 utt2 utt3
cat_b utt4 utt5
cat_c utt6
```

#### CategoryPowerSampler

CategoryPowerSampler balances categories with a power-law distribution. Use it
when you want to upsample low-resource categories without full balancing.
`min_batch_size`/`max_batch_size` bound the batch size, and
`dataset_scaling_factor` controls how aggressively samples are reused.
This sampler follows the idea in
[Scaling Speech Technology to 1,000+ Languages](https://arxiv.org/abs/2305.13516).

```yaml
dataloader:
  train:
    iter_factory:
      _target_: espnet2.iterators.category_iter_factory.CategoryIterFactory
      batch_type: catpow
      sampler_args:
        category2utt_file: ${stats_dir}/train/utt2category
        shape_files:
          - ${stats_dir}/train/feats_shape
        batch_bins: 12000000
        min_batch_size: 1
        max_batch_size: 32
        upsampling_factor: 1.0
        dataset_scaling_factor: 1.2
```

#### CategoryDatasetPowerSampler

`category_upsampling_factor` balances categories within each dataset, while
`dataset_upsampling_factor` balances across datasets. `dataset_scaling_factor`
controls overall resampling intensity, and `min_batch_size`/`max_batch_size`
bound batch size.
See also
[Scaling Speech Technology to 1,000+ Languages](https://arxiv.org/abs/2305.13516).

```yaml
dataloader:
  train:
    iter_factory:
      _target_: espnet2.iterators.category_iter_factory.CategoryIterFactory
      batch_type: catpow_balance_dataset
      sampler_args:
        category2utt_file: ${stats_dir}/train/utt2category
        dataset2utt_file: ${stats_dir}/train/dataset2utt
        utt2dataset_file: ${stats_dir}/train/utt2dataset
        shape_files:
          - ${stats_dir}/train/feats_shape
        batch_bins: 12000000
        min_batch_size: 1
        max_batch_size: 32
        category_upsampling_factor: 1.0
        dataset_upsampling_factor: 1.0
        dataset_scaling_factor: 1.2
```
