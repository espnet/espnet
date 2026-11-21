---
title: ESPnet3 Provider And Runner
author:
  name: "Masao Someki"
date: 2026-05-26
---

# ESPnet3 Provider And Runner

This page is for people who want to implement or modify a parallel workload
built on `espnet3/parallel/`.

Start with [ESPnet3 Parallel](./index.html) for the high-level flow. This page
covers the subclass contracts, the writer hooks, shard-local files, config
knobs, and resume/locking semantics.

Read the generated API docs before changing these classes:

- [`EnvironmentProvider`](../../../guide/espnet3/parallel/EnvironmentProvider.html)
  (`espnet3/parallel/env_provider.py`) — decides how runtime objects
  (dataset/model/etc.) are built.
- [`BaseRunner`](../../../guide/espnet3/parallel/BaseRunner.html)
  (`espnet3/parallel/base_runner.py`) — decides how one shard is processed,
  how outputs are written, shard planning, locking, and dispatch.

Useful concrete examples:

- [`InferenceProvider`](../../../guide/espnet3/systems/InferenceProvider.html) /
  [`InferenceRunner`](../../../guide/espnet3/systems/InferenceRunner.html)
  (`espnet3/systems/base/`) — the stage-facing pair used by `infer()`; see
  [Inference Provider](./inference_provider.html).
- [`CollectStatsInferenceProvider`](../../../guide/espnet3/components/CollectStatsInferenceProvider.html) /
  [`CollectStatsRunner`](../../../guide/espnet3/components/CollectStatsRunner.html)
  (`espnet3/components/data/collect_stats.py`).

## Minimal example

```python
from espnet3.parallel.env_provider import EnvironmentProvider
from espnet3.parallel.base_runner import BaseRunner


class MyProvider(EnvironmentProvider):
    def build_env_local(self):
        return {"dataset": build_dataset(self.config), "model": build_model(self.config)}

    def build_worker_setup_fn(self):
        config = self.config  # capture only picklable, plain values -- not self

        def setup():
            return {"dataset": build_dataset(config), "model": build_model(config)}

        return setup


class MyRunner(BaseRunner):
    @staticmethod
    def forward(idx, dataset, model, **env):
        sample = dataset[idx]
        return model(sample)


provider = MyProvider(config)
runner = MyRunner(provider, output_dir="exp/my_stage")
runner(range(len(dataset)))
```

Constraints:

- `forward()` must stay a `@staticmethod` and must not capture `self` (it has
  to be pickle-safe for Dask).
- `build_worker_setup_fn()` must return a zero-argument **function**, not the
  env dict itself; the function runs once per Dask worker.
- env keys are injected into `forward(...)`/hooks purely by parameter-name
  matching (`**env` catches whatever is left over).

## EnvironmentProvider contract

| Method | Called when | Must return |
|---|---|---|
| `build_env_local()` | Once on the driver, only for `parallel.env: local` | `dict` of env objects (e.g. `dataset`, `model`) |
| `build_worker_setup_fn()` | Once, on the driver, before dispatch; the returned callable then runs once **per Dask worker** | a zero-arg callable returning the same shape of `dict` |

[`InferenceProvider`](../../../guide/espnet3/parallel/InferenceProvider.html)
(`espnet3/parallel/inference_provider.py`) is one concrete base: it declares
`build_dataset(config)`/`build_model(config)` as abstract static methods and
implements `build_env_local`/`build_worker_setup_fn` in terms of them,
pre-building `self._local_env` once in `__init__` so local execution avoids
rebuilding on every call. This is a **different class** from
[`espnet3.systems.base.inference_provider.InferenceProvider`](../../../guide/espnet3/systems/InferenceProvider.html),
which is the one actually used by `infer()` — see
[Inference Provider](./inference_provider.html) for that one.

## BaseRunner contract

Constructor:

| Arg | Default | Meaning |
|---|---|---|
| `provider` | required | An `EnvironmentProvider` instance. |
| `batch_size` | `None` | If set, indices are chunked into lists before being passed to `forward`. |
| `output_dir` | `None` | Root directory for shard subdirectories; **required** at call time (`__call__` raises `RuntimeError` if unset). |
| `shard_subdir` | `""` | Optional subdirectory under `output_dir` (e.g. a test-set name) so multiple runs can share one `output_dir`. |
| `resume` | `True` | Skip shards whose `done` marker already exists; see [Resume and locking](#resume-and-locking) below. |

Hooks, in the order they run for one shard (`_run_one_shard`):

| Hook | Signature | Default behavior |
|---|---|---|
| `forward` (abstract) | `forward(idx, dataset, model, **env) -> Any` (`@staticmethod`) | must be implemented; no default |
| `open_writers` | `open_writers(shard_dir, **env) -> dict` (`@staticmethod`) | returns `{}` |
| `write_record` | `write_record(writers, result, state, **env) -> None` (`@staticmethod`) | appends `result` to `state["records"]` |
| `close_writers` | `close_writers(writers, state, **env) -> dict \| None` (`@staticmethod`) | closes any `.close()`-able values in `writers` |
| `merge` | `merge(self, shard_dirs) -> Any` (instance method) | returns `None` |

Lower-level hooks (`init_state`, `reduce_state`, `finalize_state`) call the
four hooks above; override them only if the state dict itself needs a
different shape.

```python
state = cls.init_state(shard_id=shard_id, **env)      # -> open_writers(...)
for item in items:
    result = cls.forward(item, **env)
    state = cls.reduce_state(state, result, shard_id=shard_id, **env)  # -> write_record(...)
cls.finalize_state(state, shard_id=shard_id, **env)    # -> close_writers(...)
```

Choose a hook by what you need:

- Only need one in-memory value per item → override `forward()` only (default
  `write_record` accumulates results in `state["records"]`).
- Need to stream results to shard-local files → override `open_writers()` /
  `write_record()` / `close_writers()`.
- Need to combine shard files into one final artifact → override `merge()`.

### Real example: InferenceRunner

[`InferenceRunner`](../../../guide/espnet3/systems/InferenceRunner.html)
(`espnet3/systems/base/inference_runner.py`) is the best in-tree reference for
the writer-style pattern: `open_writers()` prepares shard-local SCP metadata,
`write_record()` validates one result and appends to `<field>.scp`,
`close_writers()` closes handles and writes `field_keys.txt`, and `merge()`
uses [`concatenate_shard_files()`](../../../guide/espnet3/parallel/concatenate_shard_files.html)
to concatenate each field's shard fragments, in shard-id order, into the final
`<field>.scp` under `output_dir/shard_subdir`.

### prepare_model_inputs and format_output

`InferenceRunner.forward` builds the checked inputs dict, calls
`prepare_model_inputs` on it, calls the model with the result, then calls
`format_output` on whatever the model returned — before contract validation
runs. Both are `@staticmethod`s a subclass overrides independently; neither
touches contract checking itself:

```python
class ASRInferenceRunner(InferenceRunner):
    @staticmethod
    def prepare_model_inputs(inputs):
        # inputs holds already-checked Kind values (Audio for "speech" here).
        # Adapt them to the backend's own argument shape.
        inputs = dict(inputs)
        speech = inputs["speech"]
        inputs["speech"] = speech.array.T if speech.array.ndim == 2 else speech.array
        return inputs

    @staticmethod
    def format_output(result, *, batched):
        # result is whatever the backend returned (Speech2Text n-best here).
        # Return the named outputs the contract declares, before validation.
        def best(hypotheses):
            return {"text": hypotheses[0][0] or ""}

        return [best(r) for r in result] if batched else best(result)
```

- `prepare_model_inputs(inputs) -> inputs`: default is the identity. Use it
  when the backend expects a different array layout or argument shape than
  the checked `Kind` value gives you (e.g. `Audio.array` transposed to
  samples-first).
- `format_output(result, *, batched) -> result`: default is the identity.
  Use it to turn a raw backend result into the named output mapping the
  contract declares (`{"text": ...}`), handling the batched case by
  returning a list.

### Writer-hook example (fixed)

```python
from pathlib import Path
from espnet3.parallel.base_runner import BaseRunner


class MyTextRunner(BaseRunner):
    @staticmethod
    def forward(idx, dataset, model, **env):
        sample = dataset[idx]
        return {"utt_id": sample["utt_id"], "text": model(sample)}

    @staticmethod
    def open_writers(shard_dir: Path, **env):
        return {"text": (shard_dir / "text").open("w", encoding="utf-8")}

    @staticmethod
    def write_record(writers, result, state, **env):
        writers["text"].write(f'{result["utt_id"]} {result["text"]}\n')

    @staticmethod
    def close_writers(writers, state, **env):
        for handle in writers.values():
            handle.close()
        return None

    def merge(self, shard_dirs):
        out_dir = self.output_dir / self.shard_subdir if self.shard_subdir else self.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        with (out_dir / "text").open("w", encoding="utf-8") as out_f:
            for shard_dir in sorted(shard_dirs):
                part = shard_dir / "text"
                if part.exists():
                    out_f.write(part.read_text(encoding="utf-8"))
        return {}
```

`close_writers` (and every other hook) must accept the same positional/`**env`
shape the base class calls it with — `close_writers(writers, state, **env)` —
even if the override ignores `state`/`env`; dropping them raises `TypeError`
at shard-finalize time.

## Parallel config

`set_parallel(config)` (called once per stage entrypoint, e.g. `infer()`)
stores the active `parallel:` block; `BaseRunner` reads it back via
`get_parallel_config()` on every `__call__`.

| Key | Default | Meaning |
|---|---|---|
| `env` | `"local"` | `local`, `local_gpu`, `kube`, or a `dask_jobqueue` cluster name (`slurm`, `sge`, `pbs`, `lsf`, `htcondor`, `moab`, `oar`, `ssh`). |
| `n_workers` | `1` | Number of Dask workers to request, **and the number of shards `_plan_shards` divides the items into**, for every `env` including `local` — `n_workers` is read the same way regardless of `env`. |
| `options` | `{}` | Extra kwargs forwarded to the cluster constructor (`LocalCluster`, `SLURMCluster`, ...). |

```yaml
parallel:
  env: local
  n_workers: 1
```

`env: local` still runs on the local machine, but through a real Dask
`LocalCluster(n_workers=...)` — not a plain sequential loop. Raising
`n_workers` under `env: local` does parallelize shards across local Dask
worker processes. `BaseRunner.__call__` only falls back to the sequential,
non-Dask `_run_local` path when `get_parallel_config()` returns `None`,
i.e. no stage has called `set_parallel(...)` yet; every stage entrypoint
(`infer()`, `measure()`, ...) calls `set_parallel()` before running, so that
fallback is not reachable through normal stage execution.

**Dask cluster lifetime.** For any `env` (including `local`), `_run_parallel_dask`
opens the cluster with `get_client(...)` as a context manager and tears it down
(`client.close()` plus the cluster's `close()`/`shutdown()`) as soon as that
one `BaseRunner.__call__` returns. A cluster is not kept alive across calls —
a stage that calls a runner once per test set (e.g. `infer()` looping over
`config.dataset.test`) spins up and tears down a fresh cluster **per test
set**, not once for the whole stage.

## Resume and locking

Shard planning and locking live under `output_dir/shard_subdir/`:

```text
output_dir/
  shard_subdir/
    manifest.json      # shard plan: {shard_id, items} list
    split.0/
      lock              # present while a process owns this shard
      done               # written only after a full, successful pass
    split.1/
    ...
```

- On first run, the shard plan is written to `manifest.json`. On a resumed
  run (`resume=True`, the default), the newly computed plan must match the
  stored one (same number of shards, same items/order); otherwise
  `BaseRunner` raises rather than silently reprocessing different shards.
- A shard whose `done` file already exists is skipped when `resume=True`.
  Otherwise the shard directory is locked (an atomic `O_CREAT|O_EXCL` file
  create) before it runs, and unlocked in a `finally` once it finishes.

Two current caveats worth knowing before relying on resume:

- **Resume does not fingerprint the provider/model config** — only the shard
  *plan* (item indices/batching) is compared. Changing the checkpoint, beam
  size, or any other inference/provider setting and re-running with
  `resume=True` (default) will skip shards that are already marked `done` and
  silently keep the previous run's outputs. Pass `resume=False`, or remove the
  stale `output_dir`, whenever a config change should invalidate old shard
  outputs.
- **Lock acquisition is not transactional across shards.** If a shard is
  already locked by another process, `BaseRunner` raises immediately without
  releasing the locks it already acquired on other shards earlier in the same
  call. If a run is interrupted mid-lock-acquisition or crashes, you may need
  to manually remove leftover `split.N/lock` files (once you've confirmed no
  other process actually holds them) before retrying.

## Common mistakes

| Mistake | Fix |
|---|---|
| `def forward(self, idx): ...` | Keep `forward` a `@staticmethod`; never capture `self`. |
| `def build_worker_setup_fn(self): return {"model": ...}` | Return a zero-arg **function**, not the dict — otherwise setup runs on the driver, not the worker. |
| Loading the model/checkpoint inside `forward()` | Build it once in the provider (`build_env_local`/worker setup), not per item. |
| Provider env keys don't match `forward()` parameter names | Injection is by name; mismatched keys are silently dropped unless caught by `**env`, or raise `TypeError: missing required argument`. |
| Writing shard-local files but leaving `merge()` as the no-op default | Final outputs stay split across `split.N/`; override `merge()` whenever shard outputs must become one final artifact. |
| Debugging with `resume=True` after changing code/config | See [Resume and locking](#resume-and-locking) — clear `output_dir` or pass `resume=False`. |

## See also

<DocCards :cols="3">
  <DocCard
    title="ESPnet3 Parallel"
    desc="Return to the high-level parallel execution overview."
    icon="tabler:route"
    href="./index.html"
  />
  <DocCard
    title="Inference Provider"
    desc="See the inference-stage provider pattern and YAML wiring."
    icon="tabler:cpu"
    href="./inference_provider.html"
  />
  <DocCard
    title="Data Preparation"
    desc="Using runners for collect_stats and other data-side jobs."
    icon="tabler:database"
    href="./data_preparation.html"
  />
  <DocCard
    title="EnvironmentProvider API"
    desc="Read the generated contract for local and worker env setup."
    icon="tabler:book"
    href="../../../guide/espnet3/parallel/EnvironmentProvider.html"
  />
  <DocCard
    title="BaseRunner API"
    desc="Read the generated contract for forward, writer hooks, and merge."
    icon="tabler:book"
    href="../../../guide/espnet3/parallel/BaseRunner.html"
  />
  <DocCard
    title="InferenceRunner API"
    desc="Inspect the writer-style runner used by base inference."
    icon="tabler:file-code"
    href="../../../guide/espnet3/systems/InferenceRunner.html"
  />
  <DocCard
    title="X-Vector Extraction"
    desc="End-to-end concrete example: computing speaker embeddings with a custom Provider and Runner."
    icon="tabler:file-code"
    href="./xvector.html"
  />
</DocCards>
