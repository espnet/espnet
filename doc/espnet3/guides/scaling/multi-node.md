---
title: Multi-node training
author:
  name: "Masao Someki"
date: 2026-05-28
---

# Multi-node training

This guide covers scaling model training beyond a single machine — multiple
nodes, multiple GPUs per node, and the dataloader sharding that keeps them fed.

::: important
Multi-node training in ESPnet3 is configured entirely through `trainer`.
The `parallel` block is for Dask-backed runner workloads, not Lightning DDP.
Do not mix them up.
:::

## The two numbers you always set

```yaml
num_device: 8    # GPUs per node
num_nodes: 2     # number of nodes
```

These two top-level keys are used through interpolation everywhere else:

```yaml
trainer:
  accelerator: gpu
  devices: ${num_device}
  num_nodes: ${num_nodes}
  strategy: ddp
```

That is the minimal multi-node training config.
Lightning handles process launching, rank assignment, and gradient
synchronization across nodes.

## Choosing a strategy

| Strategy | When to use |
| --- | --- |
| `ddp` | Default. Gradients are all-reduced across all ranks. Works for most models. |
| `ddp_find_unused_parameters_true` | Same as `ddp` but enables `find_unused_parameters=True`. Use when your model has branches that are not always active. |
| `fsdp` | Fully Sharded Data Parallel. Use when a model is too large to fit on one GPU even with DDP. |

For most speech model recipes, `ddp` is the right choice.
Switch to `fsdp` only if you hit GPU memory limits that gradient checkpointing
does not solve.

## Dataloader sharding

When training across multiple GPUs, each GPU must see a different subset of
the data. `total_shards`/`dist_world_size` are attributes ESPnet3 reads off a
`ShardedDataset` instance (`DataLoaderBuilder._maybe_shard_dataset` calls
`getattr(dataset, "total_shards", ...)`), so they belong under `data_src_args`
in the `dataset:` block, not under `dataloader:`.

```yaml
dataset:
  train:
    - data_src: my_recipe/asr
      data_src_args:
        split: train
        total_shards: 16        # total number of shards to split the data into
        dist_world_size: 16     # must equal num_nodes * num_device
```

**Rules:**

- `dist_world_size` must equal `num_nodes` times `num_device` exactly.
  ESPnet3 validates this at the start of training and raises a `RuntimeError`
  if they differ.
- `total_shards` must be divisible by `dist_world_size`.
- A larger `total_shards` gives more fine-grained shard rotation across
  epochs. A common choice is `total_shards = dist_world_size` (one shard per
  rank) or a small multiple of it.
- Rank/world size themselves come from `torch.distributed` at runtime, not
  from any config value — they are only queried when `num_device > 1`. A
  multi-node job with exactly **one** GPU per node (`num_device: 1`,
  `num_nodes > 1`) is therefore treated as world_size=1/rank=0 for sharding
  purposes even though the real distributed world size is larger. Use
  `num_device > 1` per node, or avoid `ShardedDataset` for single-GPU-per-node
  multi-node jobs.

**Example: 2 nodes × 8 GPUs = 16 ranks**

```yaml
num_device: 8
num_nodes: 2

trainer:
  accelerator: gpu
  devices: ${num_device}
  num_nodes: ${num_nodes}
  strategy: ddp

dataset:
  train:
    - data_src: my_recipe/asr
      data_src_args:
        split: train
        total_shards: 16
        dist_world_size: 16
  valid:
    - data_src: my_recipe/asr
      data_src_args:
        split: valid
        total_shards: 16
        dist_world_size: 16

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
  valid:
    iter_factory:
      _target_: espnet2.iterators.sequence_iter_factory.SequenceIterFactory
      shuffle: false
      collate_fn: ${dataloader.collate_fn}
      batches:
        type: unsorted
        shape_files:
          - ${stats_dir}/valid/feats_shape
        batch_bins: ${dataloader.train.iter_factory.batches.batch_bins}
```

::: warning
This `iter_factory` + shape-file `batches` combination does not actually
support `total_shards > 1` today — shape files are keyed by the unsharded
dataset's index, which no longer matches after sharding. See
[Dataset sharding](./dataset-sharding.html#common-mistakes) before combining
sharding with `iter_factory`; keep `total_shards: 1` unless you control batch
construction yourself (e.g. `ChunkIterFactory` with an explicit `batches`
list), or shard through the standard `DataLoader` path instead (and set
`trainer.use_distributed_sampler: false` explicitly — see the same section).
:::

## How shard rotation works

ESPnet3 selects one shard per `(epoch, rank)` pair using:

```
shard_idx = ((epoch × world_size) + rank) % total_shards
```

This means:
- at epoch 0, rank 0 gets shard 0, rank 1 gets shard 1, ...
- at epoch 1, the rotation advances by `world_size`
- every shard is seen once per `total_shards / world_size` epochs

Setting `total_shards` to a multiple of `dist_world_size` ensures that all
shards are eventually visited. `valid` is sharded and rotated the same way as
`train` (same `epoch` value), so per-epoch `valid/loss` is not drawn from a
fixed slice of data.

## Launching multi-node jobs

Lightning multi-node training needs a launcher that starts one process per
GPU across all nodes.

**SLURM example:**

```bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8

srun python run.py --stage 5 --stop_stage 5 \
    --config conf/training.yaml
```

When using `srun`, Lightning detects the SLURM environment and sets
`MASTER_ADDR`, `MASTER_PORT`, `WORLD_SIZE`, and `LOCAL_RANK` automatically.

**Manual launch (no scheduler):**

On each node, run:

```bash
torchrun \
    --nproc_per_node=8 \
    --nnodes=2 \
    --node_rank=<0 or 1> \
    --master_addr=<IP of node 0> \
    --master_port=29500 \
    run.py --stage 5 --stop_stage 5 \
           --config conf/training.yaml
```

Replace `<0 or 1>` with the rank of the current node and `<IP of node 0>`
with the network address of your main node.

## collect_stats across multiple nodes

`collect_stats` uses the `parallel` block, not `trainer`.
It runs through the Dask-backed provider/runner path and does not follow the
same multi-node mechanism as Lightning.

For large-scale stats collection across a cluster, see
[Inference at scale](./inference.html#hpc-cluster-slurm-and-pbs) for the
equivalent `parallel.env: slurm` approach.

For single-machine stats collection before a multi-node training run, the
default `parallel.env: local` is usually sufficient.

## Common mistakes

**`dist_world_size` does not match the runtime world size.**
Set `dist_world_size` to the literal product of `num_nodes` and `num_device`
(OmegaConf has no multiply operator).
Leaving it at `1` while running multi-GPU will raise a `RuntimeError`.

**`total_shards` is not divisible by `dist_world_size`.**
For example, `total_shards: 10` with `dist_world_size: 8` will fail.
Round up `total_shards` to the nearest multiple of `dist_world_size`.

**One GPU per node.**
Rank/world size are only queried from `torch.distributed` when
`num_device > 1`; with `num_device: 1` and `num_nodes > 1`, sharding and the
NaN-skip check both fall back to world_size=1/rank=0 regardless of the real
distributed world size. Prefer more than one GPU per node when using
`ShardedDataset`.

**Putting DDP strategy settings under `parallel.options`.**
`parallel.options` is for Dask cluster options, not Lightning.
Strategy goes under `trainer`.

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Training Config"
    desc="Full schema for trainer, dataloader, optimizer, and scheduler."
    icon="tabler:settings-2"
    href="../../config/train_config.html"
  />
  <DocCard
    title="Dataset sharding"
    desc="ShardedDataset, the shard rotation formula, and known iter_factory/DDP caveats."
    icon="tabler:layout-grid"
    href="./dataset-sharding.html"
  />
  <DocCard
    title="Multi-GPU (from PyTorch)"
    desc="The two parallel layers explained for PyTorch users."
    icon="tabler:brand-python"
    href="../coming-from-pytorch/multi-gpu.html"
  />
  <DocCard
    title="Inference at scale"
    desc="Scale inference across a cluster with the provider/runner layer."
    icon="tabler:player-play"
    href="./inference.html"
  />
</DocCards>
