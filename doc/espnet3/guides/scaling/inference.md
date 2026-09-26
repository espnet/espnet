---
title: Inference at scale
author:
  name: "Masao Someki"
date: 2026-05-28
---

# Inference at scale

This guide covers running the `infer` stage across multiple GPUs or a cluster
using the provider/runner layer and the `parallel` config block.

::: important
Inference parallelism uses `parallel`, not `trainer`.
`trainer` controls Lightning DDP for model training only.
:::

## The provider/runner execution model

Inference in ESPnet3 works through a shard-and-merge pipeline:

1. The test set is split into shards.
2. Each Dask worker loads the model and dataset independently via the provider.
3. Each worker runs `forward()` over its shard and writes outputs locally.
4. The driver merges shard outputs into final SCP files.

This design means worker code is identical whether you run on one CPU, four
local GPUs, or a 32-worker SLURM cluster.
Only the `parallel` block in `inference.yaml` changes.

## Minimal inference config

```yaml
parallel:
  env: local
  n_workers: 1

input_key: speech

provider:
  _target_: espnet3.systems.base.inference_provider.InferenceProvider

runner:
  _target_: espnet3.systems.base.inference_runner.InferenceRunner
```

This runs inference sequentially on one process.
Scale it up by changing `parallel` only.

## Local multi-GPU inference

Use `local_gpu` to assign one Dask worker per visible GPU on a single machine.

```yaml
parallel:
  env: local_gpu
  n_workers: 4
```

`n_workers` must not exceed the number of visible GPUs.
ESPnet3 raises an error if it does.

The provider's `build_model()` resolves the device for each worker through
`LOCAL_RANK` and `CUDA_VISIBLE_DEVICES`, so no device assignment code is
needed in `forward()`.

## Batch inference

Set `batch_size` to pass a list of indices to `forward()` instead of one
at a time:

```yaml
batch_size: 32
```

Your runner's `forward()` must handle both forms when needed:

```python
@staticmethod
def forward(idx, dataset, model, **env):
    if isinstance(idx, int):
        samples = [dataset[idx]]
    else:
        samples = [dataset[i] for i in idx]
    return model(samples)
```

Leave `batch_size` unset (or `batch_size: null`) when your model or
`output_fn` only supports single-sample inference.

## HPC cluster: SLURM and PBS

Use a JobQueue backend to submit each Dask worker as a scheduler job.

**SLURM example:**

```yaml
parallel:
  env: slurm
  n_workers: 8
  options:
    queue: gpu
    account: my-lab
    cores: 4
    processes: 1
    memory: 32GB
    walltime: "04:00:00"
    interface: ib0
    log_directory: logs/dask
    job_script_prologue:
      - "module load cuda"
      - "source .pixi/envs/default/bin/activate"
    job_extra_directives:
      - "--gres=gpu:1"
```

This requests 8 SLURM jobs, each with one GPU.
`options` is forwarded directly to `dask_jobqueue.SLURMCluster`.

**PBS example:**

```yaml
parallel:
  env: pbs
  n_workers: 8
  options:
    queue: gpu
    cores: 4
    processes: 1
    memory: 32GB
    walltime: "04:00:00"
    job_extra_directives:
      - "-l select=1:ngpus=1"
```

The Python code — provider, runner, `forward()` — does not change between
local and cluster runs.

## Required packages per backend

Different backends need extra packages.

| Backend | Required package |
| --- | --- |
| `local` | `dask[distributed]` |
| `local_gpu` | `dask[distributed]`, `dask-cuda` |
| `slurm` / `pbs` / `sge` / `lsf` | `dask[distributed]`, `dask-jobqueue` |
| `kube` | `dask[distributed]`, `dask-kubernetes` |

## Resume

If inference is interrupted, completed shards are detected by a `done` marker
file written by the runner.
Restart the `infer` stage with `resume=True` to skip finished shards:

```yaml
runner:
  _target_: espnet3.systems.base.inference_runner.InferenceRunner
  resume: true
```

Completed `split.N/done` files are not re-run.
This means partial results from a failed shard are discarded and re-run
from scratch.

## Custom provider

Override `InferenceProvider` when the default dataset or model construction
logic does not fit your recipe.

**Override dataset construction:**

```python
from hydra.utils import instantiate

from espnet3.systems.base.inference_provider import InferenceProvider


class MyInferenceProvider(InferenceProvider):
    @staticmethod
    def build_dataset(config):
        organizer = instantiate(config.dataset)
        dataset = organizer.test[config.test_set]
        return FilteredDataset(dataset, min_duration=0.5)
```

**Override model construction:**

```python
class MyInferenceProvider(InferenceProvider):
    @staticmethod
    def build_model(config):
        model = instantiate(config.model, device="cpu")
        model.load_language_model(config.lm_path)
        model.eval()
        return model
```

Point to your provider in `inference.yaml`:

```yaml
provider:
  _target_: src.inference.MyInferenceProvider
```

## Custom runner

Override `BaseRunner` when you need custom output formats or non-SCP artifacts.

**Text output example:**

```python
from pathlib import Path

from espnet3.parallel.base_runner import BaseRunner


class MyTextRunner(BaseRunner):
    @staticmethod
    def forward(idx, dataset, model, **env):
        sample = dataset[idx]
        hyp = model(sample["speech"])
        return {"utt_id": sample["utt_id"], "text": hyp}

    @staticmethod
    def open_writers(shard_dir: Path, **env):
        return {"text": (shard_dir / "text").open("w", encoding="utf-8")}

    @staticmethod
    def write_record(writers, result, state, **env):
        writers["text"].write(f'{result["utt_id"]} {result["text"]}\n')

    @staticmethod
    def close_writers(writers):
        for handle in writers.values():
            handle.close()
        return None

    def merge(self, shard_dirs):
        out_dir = self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "text").open("w", encoding="utf-8") as out:
            for shard_dir in sorted(shard_dirs):
                path = shard_dir / "text"
                if path.exists():
                    out.write(path.read_text(encoding="utf-8"))
        return {}
```

## Switching backends without changing code

Keep per-environment configs as separate override files:

```
conf/
  inference.yaml          # base config
  parallel_local.yaml     # env: local, n_workers: 1
  parallel_gpu.yaml       # env: local_gpu, n_workers: 4
  parallel_slurm.yaml     # env: slurm, n_workers: 8, ...
```

Then merge at run time:

```bash
# local
python run.py --stage 6 --config conf/inference.yaml

# multi-GPU
python run.py --stage 6 --config conf/inference.yaml conf/parallel_gpu.yaml

# SLURM
python run.py --stage 6 --config conf/inference.yaml conf/parallel_slurm.yaml
```

This keeps the backend selection outside of the recipe code.

## Common mistakes

**Setting `n_workers` higher than visible GPUs with `local_gpu`.**
ESPnet3 raises an error immediately.
Check `CUDA_VISIBLE_DEVICES` if the count is not what you expect.

**Hard-coding a device in `build_model()`.**
`build_model()` is called once per worker.
Let the provider's default device resolution handle `LOCAL_RANK` instead of
hard-coding `cuda:0`.

**Putting large objects into `params`.**
`params` are serialized and sent to each worker.
Large tensors, open file handles, or non-serializable objects will fail or
cause large data transfer overhead.
Build them inside `build_worker_setup_fn()` instead.

**Forgetting to implement `merge()` when using writer hooks.**
If `open_writers()` / `write_record()` are overridden but `merge()` is not,
outputs stay in shard-local `split.N/` directories and are never combined.

## Related pages

<DocCards :cols="3">
  <DocCard
    title="Parallel Config"
    desc="All env values, n_workers, and backend-specific options."
    icon="tabler:settings-2"
    href="../../core/parallel/provider_runner.html"
  />
  <DocCard
    title="Inference Config"
    desc="Full schema for provider, runner, dataset, output_fn, and artifacts."
    icon="tabler:wave-sine"
    href="../../config/infer_config.html"
  />
  <DocCard
    title="Provider and Runner"
    desc="Subclass contracts, writer hooks, and shard lifecycle."
    icon="tabler:arrows-split-2"
    href="../../core/parallel/provider_runner.html"
  />
  <DocCard
    title="InferenceProvider"
    desc="The stage-facing provider — when to use the default and when to subclass."
    icon="tabler:route"
    href="../../core/parallel/inference_provider.html"
  />
  <DocCard
    title="Multi-node training"
    desc="Scale model training to multiple nodes with trainer.num_nodes."
    icon="tabler:topology-star"
    href="./multi-node.html"
  />
</DocCards>
