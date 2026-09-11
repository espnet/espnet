# Cluster and parallel

If you are coming from ESPnet2, this is one of the biggest mental-model
changes.

In ESPnet2, parallel execution is mostly orchestrated by shell scripts such as:

- `egs2/<recipe>/asr1/asr.sh`
- `egs2/<recipe>/asr1/local/data.sh`

In ESPnet3, the shell layer is much thinner.
Parallel work moves into Python through:

- `espnet3/parallel/parallel.py`
- `espnet3/parallel/env_provider.py`
- `espnet3/parallel/base_runner.py`
- `espnet3/systems/base/inference_provider.py`
- `espnet3/systems/base/inference_runner.py`

## ESPnet2: shell controls the parallelism

In ESPnet2, recipes usually expose shell variables such as:

- `nj`
- `inference_nj`
- `gpu_inference`
- `train_cmd`
- `cuda_cmd`

Then the recipe script splits inputs and fans out jobs itself.

Typical patterns in `asr.sh` are:

- `utils/split_scp.pl ...`
- `JOB=1:${_nj} ...`
- `run.pl`, `queue.pl`, `slurm.pl`

Conceptually, the script does this:

1. count input lines
2. choose `_nj`
3. split one SCP or text file into `_nj` shards
4. submit one shell job per shard
5. merge the outputs afterward

That is why ESPnet2 recipes often feel like:

- one big shell controller
- many stage-local shell loops
- one-off splitting and merging logic per stage

## Example: ESPnet2 decoding pattern

The common ESPnet2 decoding flow looks like this:

```bash
_nj=$(min "${inference_nj}" "$(<${key_file} wc -l)")
for n in $(seq "${_nj}"); do
    split_scps+=" ${_logdir}/keys.${n}.scp"
done
utils/split_scp.pl "${key_file}" ${split_scps}

${_cmd} JOB=1:"${_nj}" "${_logdir}"/asr_inference.JOB.log \
    python ... --key_file "${_logdir}"/keys.JOB.scp
```

The important point is not the exact command.
The important point is where the responsibility lives:

- the shell script decides the shard count
- the shell script writes shard files
- the shell script launches one job per shard

## Example: ESPnet2 data preparation pattern

`local/data.sh` usually follows the same style.

One shell stage does:

1. download data
2. extract data
3. run Python helpers
4. sort files
5. split train/dev/test

For example, `egs2/an4/asr1/local/data.sh`:

- downloads AN4
- runs `local/data_prep.py`
- sorts `text`, `wav.scp`, `utt2spk`
- creates `train_dev` and `train_nodev`

So in ESPnet2, "parallel or cluster behavior" is usually recipe-shell behavior.

## ESPnet3: Python owns the execution pattern

ESPnet3 moves that responsibility into Python objects.

The core split is:

- `EnvironmentProvider`: build dataset/model/tokenizer/runtime env
- `BaseRunner`: apply one static compute function over indices

This means the execution pattern is no longer:

- "split files in shell, then call Python"

It becomes:

- "describe the runtime env in Python, then let the runner execute locally or on a cluster"



## Side-by-side comparison

| Topic | ESPnet2 | ESPnet3 |
| --- | --- | --- |
| Parallel control surface | shell vars like `nj`, `inference_nj`, `train_cmd` | YAML `parallel` block plus provider/runner code |
| Work splitting | shell-side SCP splitting | Python runner over indices |
| Worker environment | implicit in shell command line and filesystem | explicit env dict from provider |
| Cluster backend | `run.pl`, `queue.pl`, `slurm.pl` wrappers | Dask client built from `parallel.env` and `parallel.options` |
| Merge behavior | shell concatenation and stage-local scripts | runner hooks such as `merge()` |
| Reuse across local/HPC | often stage-specific shell logic | same Python code can run local or Dask |

## What replaces nj and inference_nj

In ESPnet3, the closest replacement is usually:

- `parallel.n_workers`
- optional runner `batch_size`

But the meaning is slightly different.

`nj` in ESPnet2 usually means:

- "how many shell jobs should I split this file into?"

`n_workers` in ESPnet3 usually means:

- "how many worker processes should the runtime create?"

So the old and new knobs are related, but not identical.

<DocCards :cols="3">
  <DocCard
    title="Parallel Config"
    desc="See how n_workers and backend settings are expressed in YAML."
    icon="tabler:settings-2"
    href="../../core/parallel/provider_runner.html"
  />
  <DocCard
    title="Parallel Runtime"
    desc="See how ESPnet3 maps work locally or through Dask."
    icon="tabler:binary-tree-2"
    href="../../core/parallel/index.html"
  />
  <DocCard
    title="Config Diff"
    desc="See where old nj-style recipe settings usually move."
    icon="tabler:arrows-diff"
    href="./config-diff.html"
  />
</DocCards>

## What replaces run.pl / queue.pl

In ESPnet3, the backend choice is part of config.

Typical examples:

```yaml
parallel:
  env: local
  n_workers: 8
```

```yaml
parallel:
  env: local_gpu
  n_workers: 4
```

```yaml
parallel:
  env: slurm
  n_workers: 16
  options:
    queue: batch
    cores: 4
    memory: 32GB
```

So instead of changing shell wrappers, you change the `parallel` config and
keep the Python execution path the same.

<DocCards :cols="3">
  <DocCard
    title="Parallel Config"
    desc="See local, local GPU, and cluster backend examples."
    icon="tabler:server"
    href="../../core/parallel/provider_runner.html"
  />
  <DocCard
    title="Provider and Runner"
    desc="See how backend-independent work is written once in Python."
    icon="tabler:arrows-split-2"
    href="../../core/parallel/provider_runner.html"
  />
  <DocCard
    title="System and Stages"
    desc="See how stage code receives config and launches stage behavior."
    icon="tabler:hierarchy-2"
    href="../../stages/index.html"
  />
</DocCards>

## What replaces shell-side shard logic

In ESPnet3, shard logic lives in the runner layer.

That can mean:

- iterating plain indices locally
- mapping tasks through Dask
- using reducer hooks to write shard outputs
- merging shard outputs in `merge()`

The closest current examples are:

- `espnet3/systems/base/inference_runner.py`
- `espnet3/components/data/collect_stats.py`

<DocCards :cols="3">
  <DocCard
    title="Provider and Runner"
    desc="See BaseRunner hooks, worker envs, and merge behavior."
    icon="tabler:arrows-split-2"
    href="../../core/parallel/provider_runner.html"
  />
  <DocCard
    title="Inference Provider"
    desc="See the provider contract used by parallel inference."
    icon="tabler:route"
    href="../../core/parallel/inference_provider.html"
  />
  <DocCard
    title="Stats Collection"
    desc="See how collect_stats uses dataloader and runner-style execution."
    icon="tabler:gauge"
    href="../../stages/collect-stats.html"
  />
</DocCards>

## Data preparation: what changes the most

This is the place where ESPnet2 users often expect more shell.

In ESPnet2:

- `local/data.sh` is often the center of gravity
- stage logic is mostly shell + small Python helpers

In ESPnet3:

- recipe-local `dataset/builder.py` owns source preparation and build checks
- heavier inner loops can move into provider/runner code

So the mapping is roughly:

| ESPnet2 | ESPnet3 |
| --- | --- |
| `local/data.sh` | `dataset/builder.py` |
| shell stage loop | `build()` plus optional provider/runner helper |
| split files in shell | iterate indices in a runner |

<DocCards :cols="3">
  <DocCard
    title="Data Pipeline"
    desc="See how local/data.sh maps to DatasetBuilder and dataset modules."
    icon="tabler:database"
    href="./data-pipeline.html"
  />
  <DocCard
    title="Parallel Data Prep"
    desc="See when data preparation should use provider/runner execution."
    icon="tabler:download"
    href="../../core/parallel/data_preparation.html"
  />
  <DocCard
    title="Dataset Config"
    desc="See how prepared data becomes train, valid, and test config."
    icon="tabler:settings-2"
    href="../../core/components/data-organizer.html"
  />
</DocCards>

## A good migration rule

When reading an ESPnet2 recipe, ask:

1. Which part is only stage ordering?
2. Which part is only config?
3. Which part is the real per-item computation?

Then convert them like this:

1. stage ordering -> `run.py` stage list
2. config -> `training.yaml`, `inference.yaml`, `metrics.yaml`, ...
3. per-item computation -> dataset builder, provider, or runner

## When you still do not need provider/runner

Do not over-apply the abstraction.

If a step is only:

- one archive download
- one extraction
- one quick manifest rewrite

then plain `builder.py` code is often enough.

Use provider/runner when the work is actually parallel-shaped:

- many files
- many utterances
- many download targets
- one repeated compute kernel over indices


## Related pages

<DocCards :cols="3">
  <DocCard
    title="Data pipeline"
    desc="See how dataset builders and recipe-local dataset modules replace old shell prep flows."
    icon="tabler:database"
    href="./data-pipeline.html"
  />
  <DocCard
    title="Parallel overview"
    desc="Read the developer-facing provider and runner architecture."
    icon="tabler:binary-tree-2"
    href="../../core/parallel/index.html"
  />
  <DocCard
    title="Provider / Runner"
    desc="See the core contract for worker env construction and execution."
    icon="tabler:arrows-split-2"
    href="../../core/parallel/provider_runner.html"
  />
  <DocCard
    title="Parallel config"
    desc="See how local, local GPU, and HPC backends are configured in YAML."
    icon="tabler:settings-2"
    href="../../core/parallel/provider_runner.html"
  />
  <DocCard
    title="Task to system"
    desc="See how ESPnet2 task-level logic maps to ESPnet3 systems and stages."
    icon="tabler:hierarchy-2"
    href="./task-to-system.html"
  />
</DocCards>
