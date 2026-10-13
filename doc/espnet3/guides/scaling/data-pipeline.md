# Dynamic batching and batch distribution

ESPnet3 configures training and validation loaders under `dataloader.train`
and `dataloader.valid`. SequenceIterFactory supports ESPnet batch samplers;
`iter_factory: null` selects a standard PyTorch DataLoader.

## Dynamic batching with batch_bins

The sampler determines what a batch contains. With `type: length`,
`batch_bins` counts feature lengths; with `type: numel`, it counts feature
elements, including dimensions and the sampler's padding policy. It is not a
GPU-memory limit, and a long sample or minimum batch size may exceed the budget.
`type: unsorted` uses `batch_size` instead of dynamic `batch_bins`.

Use `collect_stats` shape files for the same dataset and index ordering:

```yaml
dataloader:
  train:
    batch_distribution: per_gpu
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: true
      collate_fn: ${dataloader.collate_fn}
      batches:
        type: numel
        shape_files:
          - ${stats_dir}/train/feats_shape
        batch_bins: 4000000
        batch_size: 1
        min_batch_size: 1
```

Tune the budget against the model, sequence-length distribution, and hardware.
Changing GPU count or gradient accumulation is part of changing the experiment.

## Per-GPU and global modes

The distribution setting belongs beside `iter_factory`, not inside the
sampler's `batches` block. Set it independently for train and valid.

| Mode | Distribution over W ranks | Meaning of batch_bins |
| --- | --- | --- |
| `per_gpu` (default) | Rank r receives complete batches `batches[r::W]` | Per-rank budget; aggregate budget per step scales approximately with W |
| `global` | Rank r receives `batch[r::W]` from every batch | One sampler budget shared across ranks, as in ESPnet2 sequence batching |

On one rank, both modes yield the same batches. For a five-item batch and
two ranks, global mode yields three items on rank 0 and two on rank 1,
preserving all five without duplication. Per-GPU mode assigns that complete
five-item batch to one rank and a different complete batch to the other.

Global mode does not drop tail batches to make the number of batches divisible
by W: each rank processes every original batch. Per-GPU mode retains the
existing tail-batch alignment so all ranks have equal step counts.
Neither mode automatically rescales sampler budgets or accumulation.

Omitting the key preserves existing ESPnet3 configs. To opt in, set
`dataloader.train.batch_distribution: global` and, when desired,
`dataloader.valid.batch_distribution: global`. Keep the same global budget
when changing world size, and ensure every batch still contains at least W
items. Budget and mode are logged when the iterator is constructed.

## Safety requirements for global mode

- Use the standard SequenceIterFactory and the same complete dataset on all ranks.
- Every generated batch must contain at least `world_size` items. Smaller
  batches raise a clear error; no implicit dropping, padding, or duplication occurs.
- Do not use rank-specific dataset shards (`total_shards > 1`), chunk/category
  iterator factories, or custom factories that alter the supplied batch plan.
- Leave `shuffle_within_batch` false. That option regroups local samples after
  splitting and would change global batch membership.
- Multi-device jobs require an initialized torch.distributed process group.
  Global mode uses its actual world size, including multiple nodes.
- All ranks must use the same generated batches and iteration settings. Global
  mode compares batch-plan fingerprints and rejects differences before iteration.

Ordinary SequenceIterFactory batch-order shuffling remains supported: its shared
seed and epoch keep the global batch order aligned across ranks.

## Dataset sharding and standard DataLoader

Dataset-level sharding is separate from batch distribution. The current builder
reads `total_shards` and `dist_world_size` from the wrapped dataset, validates
them against the runtime, and selects an epoch/rank-specific shard. Configure
those dataset attributes through `data_src_args`, not dataloader keys.
Such rank-specific shards cannot be used with global mode.

Standard DataLoader configs keep their existing behavior and accept an explicit
`batch_distribution: per_gpu`. Global mode is rejected on this path because
custom samplers and Lightning sampler replacement do not provide the shared
sequence-batch contract required for element-wise splitting.

## Validation and inference

Select validation mode independently and use validation shape files, not training
shape files. Global validation has the same minimum-batch and shared-dataset rules.

Inference and stats collection use Provider/Runner worker parallelism under
`parallel`, independently of training's Lightning process group. This option
does not change inference batching or helper worker counts.

See [the migration guide](../migrating-from-espnet2/config-diff.md) for the
ESPnet2/ESPnet3 configuration mapping.
