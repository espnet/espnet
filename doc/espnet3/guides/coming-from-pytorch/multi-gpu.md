# Multi-GPU

If you come from plain PyTorch, multi-GPU code in ESPnet3 usually feels easier
because you do not need to write the distributed orchestration yourself.

The main rule is:

::: important
ESPnet3 has two different parallel layers.

- model training parallelism -> `trainer`
- runner/provider parallelism -> `parallel`
:::

Do not mix them up.

## 1. Multi-GPU training

If your goal is:

- train one model on multiple GPUs
- use DDP / FSDP / multi-node Lightning strategies

then the main config is `trainer`.

Typical example:

```yaml
num_device: 4
num_nodes: 1

trainer:
  accelerator: gpu
  devices: ${num_device}
  num_nodes: ${num_nodes}
  strategy: ddp
```

That is already the most common multi-GPU training path.

If you know raw PyTorch, this replaces a lot of manual work such as:

- process launching
- DDP setup
- rank/world-size handling
- trainer loop wiring

<DocCards :cols="3">
  <DocCard
    title="Training Config"
    desc="See trainer.devices, num_nodes, strategy, precision, and accelerator settings."
    icon="tabler:settings-2"
    href="../../config/train_config.html"
  />
  <DocCard
    title="Trainer"
    desc="See how ESPnet3 delegates training orchestration to Lightning."
    icon="tabler:bolt"
    href="../../core/components/trainer.html"
  />
  <DocCard
    title="Training Loop"
    desc="See what replaces a hand-written PyTorch training loop."
    icon="tabler:player-play"
    href="../finetuning/training-loop.html"
  />
</DocCards>

## 2. Multi-GPU or multi-worker helper execution

If your goal is instead:

- parallel inference
- runner-based preprocessing
- provider/runner workloads
- Dask-backed helper execution

then the main config is `parallel`.

Typical local multi-GPU example:

```yaml
parallel:
  env: local_gpu
  n_workers: 4
```

This means:

- use one Dask worker per visible GPU
- keep the Python compute path the same
- let ESPnet3 handle the worker setup

<DocCards :cols="3">
  <DocCard
    title="Parallel Config"
    desc="See local_gpu, n_workers, and backend options."
    icon="tabler:binary-tree-2"
    href="../../core/parallel/provider_runner.html"
  />
  <DocCard
    title="Provider and Runner"
    desc="See how worker envs and repeated compute functions are defined."
    icon="tabler:arrows-split-2"
    href="../../core/parallel/provider_runner.html"
  />
  <DocCard
    title="Inference Provider"
    desc="See the provider pattern used by parallel inference."
    icon="tabler:route"
    href="../../core/parallel/inference_provider.html"
  />
</DocCards>

## Why this is simpler than hand-written multiprocessing

In raw PyTorch code, people often write:

- `torch.multiprocessing`
- manual worker setup
- custom queue logic
- device assignment code
- ad-hoc cluster wrappers

In ESPnet3, you usually do not need to write that by hand.

For training:

- Lightning handles process orchestration

For runner/provider workloads:

- ESPnet3 handles env construction and worker execution

So the main job becomes config, not orchestration code.

## A small training example

For multi-GPU training on one node:

```yaml
trainer:
  accelerator: gpu
  devices: 4
  strategy: ddp
```

For multi-node training:

```yaml
trainer:
  accelerator: gpu
  devices: 8
  num_nodes: 2
  strategy: ddp
```

This is the part that replaces a lot of manual DDP launcher logic.

## A small provider/runner example

Suppose you have a runner-based workload and want one worker per local GPU.

```yaml
parallel:
  env: local_gpu
  n_workers: 2
```

Then your code can stay conceptually simple:

```python
provider = MyProvider(cfg)
runner = MyRunner(provider)
runner(range(len(dataset)))
```

ESPnet3 handles:

- worker startup
- env setup on each worker
- dispatch of `forward(idx, **env)`

So you do not need to write your own multi-process wrapper first.

## Local CPU vs local GPU vs cluster

The three common patterns are:

### CPU-only local helper execution

```yaml
parallel:
  env: local
  n_workers: 8
  options:
    threads_per_worker: 1
```

### Multi-GPU local helper execution

```yaml
parallel:
  env: local_gpu
  n_workers: 4
```

### HPC-backed helper execution

```yaml
parallel:
  env: slurm
  n_workers: 16
  options:
    queue: gpu
    cores: 4
    memory: 32GB
    walltime: "04:00:00"
```

The Python code can stay the same across all three.
That is one of the main benefits.

<DocCards :cols="3">
  <DocCard
    title="Cluster Migration"
    desc="See what replaces nj, run.pl, queue.pl, and shell scheduler options."
    icon="tabler:server"
    href="../migrating-from-espnet2/cluster-and-parallel.html"
  />
  <DocCard
    title="Parallel Runtime"
    desc="See how local and Dask execution share the same runner path."
    icon="tabler:binary-tree-2"
    href="../../core/parallel/index.html"
  />
  <DocCard
    title="Parallel Data Prep"
    desc="See when helper work should use provider/runner execution."
    icon="tabler:download"
    href="../../core/parallel/data_preparation.html"
  />
</DocCards>

## A common confusion

This is the confusion most PyTorch users hit first:

### trainer.devices

Use this for model training.

### parallel.n_workers

Use this for runner/provider workloads.

These are related to parallelism, but they do not control the same mechanism.

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Training Config"
    desc="See where multi-GPU training is configured through Lightning."
    icon="tabler:settings-2"
    href="../../config/train_config.html"
  />
  <DocCard
    title="Parallel Config"
    desc="See how `local`, `local_gpu`, and HPC backends are configured."
    icon="tabler:binary-tree-2"
    href="../../core/parallel/provider_runner.html"
  />
  <DocCard
    title="Provider / Runner"
    desc="Read the execution contract behind runner-based parallel workloads."
    icon="tabler:arrows-split-2"
    href="../../core/parallel/provider_runner.html"
  />
  <DocCard
    title="Inference Provider"
    desc="See the stage-facing example of dataset/model env construction."
    icon="tabler:wave-sine"
    href="../../core/parallel/inference_provider.html"
  />
  <DocCard
    title="Model and system"
    desc="See how custom models and custom stage flows fit into recipes."
    icon="tabler:puzzle"
    href="./model-and-system.html"
  />
</DocCards>
