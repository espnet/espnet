---
title: ESPnet3 Parallel
author:
  name: "Masao Someki"
date: 2026-05-26
---

# ESPnet3 Parallel

`espnet3/parallel/` is the shared provider/runner layer ESPnet3 uses to run
one piece of work (one dataset item, one shard) in parallel across a local
process, a local GPU, or a Dask-backed cluster (`slurm`, `sge`, `kube`, ...),
without the stage code itself knowing which backend is active.

The pattern has two halves:

- a **Provider** (`EnvironmentProvider`) builds the runtime objects a job
  needs once — a dataset, a model, a device — and exposes them the same way
  whether execution ends up local or distributed.
- a **Runner** (`BaseRunner`) takes a provider, plans work into shards,
  dispatches `forward()` over each shard's items, and merges shard outputs
  into one final result.

```text
EnvironmentProvider          BaseRunner
  build_env_local()  ---->   forward(idx, **env)
  build_worker_setup_fn()    open_writers / write_record / close_writers
                             merge(shard_dirs)
```

Stage code only ever constructs a provider, constructs a runner around it,
and calls the runner — shard planning, resume/locking, and the local-vs-Dask
dispatch all live inside `BaseRunner`.

## Interactive guide

### Components overview

<SectionOverview />

### Build your parallel config

<SectionConfigBuilder />

### Execution modes compared

<SectionExecutionModes />

### EnvironmentProvider timing

<SectionProvider />

### Shard lifecycle walkthrough

<SectionShardLifecycle />

## Where this is used today

- `infer()` uses
  [`InferenceProvider`](../../../guide/espnet3/systems/InferenceProvider.html) /
  [`InferenceRunner`](../../../guide/espnet3/systems/InferenceRunner.html) —
  see [Inference Provider](./inference_provider.html).
- `collect_stats` uses
  [`CollectStatsInferenceProvider`](../../../guide/espnet3/components/CollectStatsInferenceProvider.html) /
  [`CollectStatsRunner`](../../../guide/espnet3/components/CollectStatsRunner.html) —
  see [Data Preparation](./data_preparation.html).
- The LibriTTS VITS recipe uses a custom provider/runner pair to extract
  x-vectors — see [X-Vector Extraction](./xvector.html) for a full worked
  example.

## Where to go next

<DocCards :cols="3">
  <DocCard
    title="Provider / Runner"
    desc="The base contracts: env construction, forward, writer hooks, parallel config, resume/locking."
    icon="tabler:arrows-split-2"
    href="./provider_runner.html"
  />
  <DocCard
    title="Inference Provider"
    desc="The stage-facing provider/runner pair used by infer()."
    icon="tabler:wave-sine"
    href="./inference_provider.html"
  />
  <DocCard
    title="Data Preparation"
    desc="Using the same pattern for dataset preparation and collect_stats."
    icon="tabler:database"
    href="./data_preparation.html"
  />
  <DocCard
    title="X-Vector Extraction"
    desc="A full worked example: a custom Provider and Runner for speaker embeddings."
    icon="tabler:file-code"
    href="./xvector.html"
  />
</DocCards>
