# `espnet3/parallel/`

See [`.agent/CLAUDE.md`](../../CLAUDE.md) for cross-cutting guidance.

```
espnet3/parallel/
├── base_runner.py            # BaseRunner(ABC) -- shard planning, locking, dispatch
├── env_provider.py           # EnvironmentProvider(ABC) -- "what one worker needs to run one item"
├── inference_provider.py     # a second, standalone InferenceProvider (distinct from systems/base's)
└── parallel.py               # set_parallel/get_parallel_config (module-global context), build_client/get_client (Dask)
```

The execution model shared by every stage that fans work out across items (`infer`, `collect_stats`,
`remove_long_short` -- see [`.agent/espnet3/systems/CLAUDE.md`](../systems/CLAUDE.md) and
[`.agent/espnet3/components/CLAUDE.md`](../components/CLAUDE.md) for the callers).

- **`base_runner.py`** -- `BaseRunner(ABC)`: shard planning, per-shard locking, dispatch (local /
  Dask), and `concatenate_shard_files` for merging shard outputs back into one file. Subclassed by
  `InferenceRunner`, `CollectStatsRunner`, `RemoveLongShortRunner`.
- **`env_provider.py`** -- `EnvironmentProvider(ABC)`: the contract describing what state one worker
  needs to process one item (model/dataset construction, device placement).
- **`inference_provider.py`** -- a second `InferenceProvider(EnvironmentProvider)` implementation,
  distinct from `systems/base/inference_provider.py`'s class of the same name -- check which one a
  given call site actually imports before changing either.
- **`parallel.py`** -- `set_parallel`/`get_parallel_config` (the module-global "current parallel
  context", read by the `Provider`s above), `build_client`/`get_client` (constructs the Dask `Client`
  for `parallel.env: local` vs. a real cluster backend), `DictReturnWorkerPlugin`/
  `wrap_func_with_worker_env` (propagates the env context into Dask worker processes).

Naming convention when adding a new parallel stage: a `*Runner`/`*Provider` pair, named the same way
as the existing ones (`InferenceRunner`/`InferenceProvider`,
`CollectStatsRunner`/`CollectStatsInferenceProvider`,
`RemoveLongShortRunner`/`RemoveLongShortProvider`) -- see `.agent/CLAUDE.md`'s naming conventions
section.

## Recommended usage

Use the shared `Runner`/`Provider` abstraction for recipe or system stages that process many
independent items. It keeps the stage usable across a local single-process run and configured
parallel backends without making the recipe own scheduler-specific dispatch code.

The usual flow is:

1. A `Runner` partitions the work and manages dispatch and shard output.
2. A `Provider` constructs the worker-local model/dataset environment and processes one item.
3. The runner merges shard outputs into the stage's final artifact.

A minimal configuration is:

```yaml
parallel:
  env: local
  n_workers: 1
```

Keep the stage-specific worker arguments in the Provider, and keep shard planning, dispatch, and
concatenation in the Runner. For a complete implementation, read
`espnet3/systems/tts/remove_long_short_runner.py` and
`espnet3/systems/tts/remove_long_short_provider.py`. For model-backed inference, compare
`espnet3/systems/base/inference_runner.py` and
`espnet3/systems/base/inference_provider.py`. Recipe-level configuration and call sites are also
available under `egs3/mini_an4/asr/` and `egs3/librispeech_100/asr/`.

When adding a new parallel stage, start from one of those pairs, write a focused unit test for the
Runner and Provider contracts, and add a small integration invocation when the stage is part of a
recipe workflow. Prefer this abstraction over direct multiprocessing, Dask, or Slurm calls in a
recipe so the same code remains seamless across development, CI, and cluster environments.
