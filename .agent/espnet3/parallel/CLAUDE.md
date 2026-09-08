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

> This is the design review's known-issues theme F: the Runner/Provider abstraction leaks state (a
> flat `**env` namespace shared between runner bookkeeping and provider params, `self`-capturing
> closures that pickle more than intended, per-call Dask cluster lifecycle). Read `.agent/CLAUDE.md`
> section 2 and `espnet3_fable_review.md` before making non-trivial changes here, and check which of
> the two `InferenceProvider` classes you're actually touching.

Naming convention when adding a new parallel stage: a `*Runner`/`*Provider` pair, named the same way
as the existing ones (`InferenceRunner`/`InferenceProvider`,
`CollectStatsRunner`/`CollectStatsInferenceProvider`,
`RemoveLongShortRunner`/`RemoveLongShortProvider`) -- see `.agent/CLAUDE.md`'s naming conventions
section.
