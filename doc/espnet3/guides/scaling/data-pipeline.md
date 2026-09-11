---
title: Large-scale data
author:
  name: "Masao Someki"
date: 2026-05-28
---

# Large-scale data

This guide covers the dataloader settings that matter most when training on
large datasets — sharding, dynamic batching, and the relationship between
data pipeline config and multi-GPU runs.

## Two paths through the dataloader

ESPnet3 supports two iteration modes.
Both are configured under `dataloader.train` and `dataloader.valid`.

| Mode | When to use |
| --- | --- |
| `iter_factory` (ESPnet-style) | Dynamic batching by frame/token count, built-in sharding, shuffle with reproducibility |
| Standard DataLoader (`iter_factory: null`) | Fixed batch size, custom sampler, PyTorch-native iteration |

For large speech datasets, `iter_factory` with `batch_bins` is almost always
the right choice.

## Dynamic batching with batch_bins

Fixed batch sizes waste GPU memory because sequences vary in length.
`batch_bins` packs samples until the total frame count hits a target, giving
roughly equal compute per batch regardless of sequence length.

```yaml
dataloader:
  train:
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: true
      collate_fn: ${dataloader.collate_fn}
      batches:
        type: unsorted
        shape_files:
          - ${stats_dir}/train/feats_shape
        batch_bins: 4000000
```

`batch_bins` is measured in frames (or tokens, depending on the feature type).
`shape_files` must point to the output of `collect_stats`, which writes
per-utterance frame counts to `${stats_dir}/train/feats_shape`.

**Choosing batch_bins:**

A rough starting point is to divide your target GPU memory budget by the
memory cost of one frame.
For 40 GB GPUs training on 80-dimensional filterbanks:

- `batch_bins: 2000000` → conservative, leaves room for long sequences
- `batch_bins: 4000000` → typical
- `batch_bins: 8000000` → aggressive; monitor for OOM on long-tail sequences

Increase `batch_bins` as long as GPU utilization rises without OOM errors.
Long-tail sequences set the effective ceiling — if one utterance is unusually
long, the batch it lands in may have only one sample.

## Sharding for multi-GPU and multi-node

Each GPU needs a non-overlapping slice of the training data. This is a
**dataset-level** attribute, not a dataloader key: `total_shards`/
`dist_world_size` are read off a `ShardedDataset` instance
(`getattr(dataset, "total_shards", ...)`), so set them through `data_src_args`
in the `dataset:` block -- **not** under `dataloader:`. Those keys are not
read by `DataLoaderBuilder`; with the standard-DataLoader path
(`iter_factory: null`) leaving them under `dataloader:` raises a `TypeError`
from `torch.utils.data.DataLoader`, and with `iter_factory` set they are
silently ignored.

```yaml
dataset:
  train:
    - data_src: my_recipe/asr
      data_src_args:
        split: train
        total_shards: 16
        dist_world_size: 16
```

`dist_world_size` must equal `num_nodes` times `num_device` (write the product
as a literal -- OmegaConf has no multiply operator).
`total_shards` must be divisible by `dist_world_size`.

For single-GPU runs, either skip `ShardedDataset` entirely or set both to `1`.

::: warning
Sharding does not currently work together with the `iter_factory` +
`batch_bins` path fed from `collect_stats` shape files: those files are keyed
by the *unsharded* dataset's index, which no longer lines up once
`dataset.shard()` reindexes the data. Use `total_shards: 1` with
`iter_factory`, or use the standard `DataLoader` (`iter_factory: null`) when
sharding is required.
:::

See [Dataset sharding](./dataset-sharding.html) for the full rules and shard
rotation formula.

## Validation dataloader

The validation dataloader does not need shuffle. Give the `valid` dataset
entry the same `total_shards`/`dist_world_size` as `train` (again via
`data_src_args`) so each GPU validates only its own slice:

```yaml
dataloader:
  valid:
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: false
      collate_fn: ${dataloader.collate_fn}
      batches:
        type: ${dataloader.train.iter_factory.batches.type}
        shape_files:
          - ${stats_dir}/valid/feats_shape
        batch_bins: ${dataloader.train.iter_factory.batches.batch_bins}
```

Validation shards rotate every epoch exactly like training (both use the same
`epoch` value), so `valid/loss` is not drawn from a fixed slice across epochs
when sharding is on.

## Standard DataLoader path

For recipes that do not need dynamic batching or ESPnet-style sharding,
set `iter_factory: null` and use PyTorch DataLoader fields directly:

```yaml
dataloader:
  train:
    iter_factory: null
    batch_size: 32
    num_workers: 4
    shuffle: true
```

Custom samplers are also supported at the top level of the `dataloader` block:

```yaml
dataloader:
  sampler:
    _target_: torch.utils.data.WeightedRandomSampler
    num_samples: 10000
    replacement: true
  train:
    iter_factory: null
    batch_size: 32
```

`sampler` and `batch_sampler` are mutually exclusive.

## Collecting stats for large datasets

`collect_stats` writes the `feats_shape` files that `batch_bins` depends on.
It runs through the Dask-backed `parallel` layer and can be parallelized
independently from training:

```yaml
parallel:
  env: local
  n_workers: 8
  options:
    threads_per_worker: 1
```

For very large datasets on a cluster, use a JobQueue backend instead:

```yaml
parallel:
  env: slurm
  n_workers: 16
  options:
    queue: cpu
    cores: 4
    memory: 16GB
    walltime: "02:00:00"
```

`n_workers` here controls how many Dask workers run stats collection
concurrently — it is independent from `trainer.devices`.

**Keying and resume.** Shape/stats files are keyed by the `CombinedDataset`
global integer index (`uid=str(idx)`) for that split, not by a stable
utterance ID — reordering or changing the `dataset:` entries between
`collect_stats` and `train` silently re-maps shape entries to different
utterances, with no error. `collect_stats` also resumes by shard (a
`split.N/done` marker) without fingerprinting the model or dataset config, so
changing frontend settings (e.g. `n_mels`, `hop_length`) or dataset contents
without also changing/clearing `${stats_dir}` can skip shards and merge stale
stats. Use a fresh `stats_dir` (or delete the old one) whenever the
model/dataset config changes.

## Common mistakes

**Leaving `dist_world_size: 1` for a multi-GPU run.**
The dataloader will raise a `RuntimeError` at the start of training because
the configured `dist_world_size` (1) does not match the actual distributed
world size (number of GPUs).

**Setting `total_shards` to a value not divisible by `dist_world_size`.**
For example, `total_shards: 12` with `dist_world_size: 8` will fail.
Round `total_shards` up to the nearest multiple.

**Using `iter_factory` without running `collect_stats` first.**
`batch_bins` batching reads `shape_files` written by `collect_stats`.
If those files do not exist, the dataloader build will fail.

**Setting `batch_bins` too high for your dataset's longest sequences.**
A single very long utterance forces a batch with only that sample, which
reduces GPU utilization for that step.
Filter extreme outliers from training data or cap sequence length if this
becomes a bottleneck.

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Multi-node training"
    desc="Set trainer.num_nodes and dataloader sharding together."
    icon="tabler:topology-star"
    href="./multi-node.html"
  />
  <DocCard
    title="Dataloader Config"
    desc="Full reference for iter_factory, collate_fn, and batch strategy."
    icon="tabler:database"
    href="../../core/components/dataloader.html"
  />
  <DocCard
    title="Stats collection"
    desc="See how collect_stats writes the shape files used by batch_bins."
    icon="tabler:gauge"
    href="../../stages/collect-stats.html"
  />
</DocCards>
