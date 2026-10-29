# Parallel execution in ESPnet3

ESPnet3 has two separate execution layers: Dask runners for independent
shards (for example statistics and inference), and Lightning for model
training across devices/nodes. A Dask GPU worker count does not configure
Lightning DDP training.

## Dask configuration

A stage passes its top-level `parallel` block to `set_parallel()`:

```yaml
parallel:
  env: local
  n_workers: 2
  options:
    threads_per_worker: 1
```

`env: local` creates `distributed.LocalCluster`. `options` are forwarded to
the cluster constructor; `n_workers` controls worker/shard planning. A
configured local cluster still uses Dask even with one worker. `BaseRunner`
uses its in-process path only when no global parallel config is set.
Calling `set_parallel(None)` retains a previous global config if one exists,
so pass an explicit local config when switching execution policy.

Local GPU inference uses `LocalCUDACluster` and requires `dask_cuda` and its
CUDA dependencies:

```yaml
parallel:
  env: local_gpu
  n_workers: 1
  options:
    threads_per_worker: 1
```

The worker count cannot exceed visible GPUs. Whether the backend actually
uses CUDA is determined by the provider/model configuration as well as the
cluster's device visibility.

For a site using SLURM and `dask_jobqueue`:

```yaml
parallel:
  env: slurm
  n_workers: 4
  options:
    queue: cpu
    cores: 4
    processes: 1
    memory: 8GB
    walltime: "00:30:00"
```

Use your site's account/partition/resource settings in `options`. For GPU
workers, request a GPU in the site's `job_extra_directives` and select the
appropriate queue; the example above requests CPU jobs. The implementation
also maps supported jobqueue environments and offers `kube` through
`dask_kubernetes`; these optional packages must be installed.

## Providers, runners, and batches

An `EnvironmentProvider` implements worker setup, commonly building the
model and dataset once per worker. A `BaseRunner` subclass supplies
`forward()` plus writer/reduction/merge hooks. `InferenceRunner` already
implements persistence; keep it when a backend returns compatible named
outputs, and override formatting/input-adaptation hooks only when needed.

```python
from espnet3.parallel.parallel import get_client
from omegaconf import OmegaConf


def square(value):
    return value * value


if __name__ == "__main__":
    config = OmegaConf.create(
        {"env": "local", "n_workers": 2, "options": {"threads_per_worker": 1}}
    )
    with get_client(config) as client:
        futures = client.map(square, [1, 2, 3])
        assert client.gather(futures) == [1, 4, 9]
```

`get_client()` is a context manager that closes its client and cluster. The
`__main__` guard is needed for spawned workers. This low-level Dask example
is separate from calling a stage's configured runner.

A runner's `batch_size` groups dataset indices **before** shard planning:
`null` sends single indices and a positive integer sends index lists. The
backend must support the corresponding argument shape. In inference,
consecutive samples become list-valued backend inputs; there is no automatic
single-item retry for a backend that rejects batches.

## Outputs and resume

A file-writing runner requires `output_dir`. Inference produces
`<inference_dir>/<test_name>/split.<shard>/` with shard files/artifacts, then
merges `<field>.scp` files under the test-set directory. See
[the inference configuration](./config/infer_config.html).

`resume=True` is a runner constructor default. Its manifest records the
indices/batches and shard plan; a different plan is rejected. Completed
shards are reused. The reviewed baseline does not validate model/data
content identity, so changing weights or manifests requires a fresh output
directory or an explicit clean recomputation. It does not promise safe
concurrent writers to one output directory.

`parallel.allow_overwrite_lock: true` can replace a stale lock after its
owning process has stopped. Do not enable it while a writer is active.
Worker startup and output paths must be accessible on the selected cluster;
shared storage is usually needed for merged shard artifacts.

## Lightning training

Training resource settings belong to `num_device`, `num_nodes`, and
`trainer`, independently of Dask's `parallel.n_workers`:

```yaml
num_device: 2
num_nodes: 1

trainer:
  accelerator: gpu
  devices: ${num_device}
  num_nodes: ${num_nodes}
  strategy: ddp
  max_epochs: 10
```

Launch one coordinated training job with the site's supported Lightning
launch environment. Do not submit multiple independent `run.py` processes
against the same output directory expecting the launcher to deduplicate
non-training stages: on the reviewed baseline only bake/snapshot writes are
rank-gated, not all stage bodies. Prepare shared dataset/tokenizer/statistics
artifacts once before a distributed training run.

Dataloader sharding settings describe the underlying `ShardedDataset`, not
Dask worker count. Set `total_shards` and `dist_world_size` as dataset
attributes (typically through `data_src_args`); placing them under
`dataloader.train` does not configure dataset sharding. Match the actual
runtime topology. See the
[training configuration](./config/train_config.html#sharding-keys).

The implementation references are `espnet3/parallel/base_runner.py`,
`espnet3/parallel/env_provider.py`, `espnet3/parallel/parallel.py`,
`espnet3/components/trainers/trainer.py`, and
`espnet3/components/data/dataloader.py`.
