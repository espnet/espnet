# `espnet3/components/`

See [`.agent/CLAUDE.md`](../../CLAUDE.md) for cross-cutting guidance.

```
espnet3/components/
├── data/                     # dataset abstraction, sharding, collect_stats
│   ├── data_organizer.py     # DataOrganizer, DatasetConfig
│   ├── dataset.py            # CombinedDataset, DatasetWithTransform, ShardedDataset(ABC)
│   ├── dataset_builder.py    # DatasetBuilder(ABC) -- contract every recipe's dataset/builder.py implements
│   ├── dataset_module.py     # data_src resolution: load_dataset_module, resolve_dataset_module_name, ...
│   ├── dataloader.py         # DataLoaderBuilder
│   ├── iterator.py           # EpochSyncIterator
│   └── collect_stats.py      # collect_stats_batch / collect_stats, CollectStatsRunner + Provider
├── modeling/
│   ├── lightning_module.py   # ESPnetLightningModule -- training_step/validation_step, both optimizer paths
│   └── optimization_spec.py  # OptimizerSpec, SchedulerSpec, OptimizationStep, OptimizerRuntimeState
├── trainers/trainer.py       # ESPnet3LightningTrainer -- builds lightning.Trainer from the `trainer:` config
├── callbacks/
│   ├── default_callbacks.py  # AverageCheckpointsCallback, MetricsLogger, get_default_callbacks()
│   ├── ema.py                # EMACallback
│   └── vendored_ema.py       # EMA module (vendored from lucidrains/ema-pytorch)
├── metrics/base_metric.py    # BaseMetric(ABC) -- the contract asr's CER/WER/TER implement
└── optimizers/                # placeholder package today -- no custom classes yet; optimizers are wired
                                # directly via Hydra `_target_: torch.optim.*` in training.yaml
```

Reusable building blocks the systems in
[`.agent/espnet3/systems/CLAUDE.md`](../systems/CLAUDE.md) are assembled from. Nothing in
`components/` knows about "stages" by name -- that concept lives in `systems/` and
`utils/stages_utils.py` (see [`.agent/espnet3/utils/CLAUDE.md`](../utils/CLAUDE.md)).

## `data/`

- **`data_organizer.py`** -- `DataOrganizer`: instantiates and composes the `train`/`valid`/`test`
  dataset entries of `training_config.dataset` (each entry references a dataset via `data_src`) plus
  the shared `preprocessor:` block. `DatasetConfig` is a typed-config helper.
- **`dataset.py`** -- `CombinedDataset` (concatenates the per-split sub-datasets behind one
  `__getitem__`/`__len__`; supports both integer and string ("uid") indexing and epoch-aware
  sharding via `.shard()`), `DatasetWithTransform` (applies a per-sample `transform:` before
  preprocessing), `ShardedDataset(ABC)` (the sharding contract).
- **`dataset_builder.py`** -- `DatasetBuilder(ABC)`: the contract every recipe's
  `dataset/builder.py` implements (`is_source_prepared`, `prepare_source`, `is_built`, `build`).
  See [`.agent/egs3/CLAUDE.md`](../../egs3/CLAUDE.md) for two real implementations to copy from when
  creating a new recipe.
- **`dataset_module.py`** -- resolves a dataset reference (`data_src: "<recipe>/<task>"`, a
  dotted module path, or the recipe's own local `dataset/__init__.py`) to a `DatasetBuilder`/`Dataset`
  pair: `load_dataset_module`, `resolve_dataset_module_name`, `parse_dataset_reference_config`,
  `instantiate_dataset_reference`.
- **`dataloader.py`** -- `DataLoaderBuilder`: turns a `dataloader:` config block + a
  `CombinedDataset` into the iterator/DataLoader Lightning actually iterates over (handles the
  `total_shards`/`dist_world_size` shard rotation and the `iter_factory` vs. plain-`DataLoader` split).
- **`iterator.py`** -- `EpochSyncIterator`: keeps the train/valid iterator epoch counters in
  sync with Lightning's `current_epoch`.
- **`collect_stats.py`** -- `collect_stats_batch` / `collect_stats(config)`: the `collect_stats`
  stage's implementation, plus `CollectStatsRunner`/`CollectStatsInferenceProvider`
  (a `parallel.BaseRunner`/`EnvironmentProvider` pair -- see
  [`.agent/espnet3/parallel/CLAUDE.md`](../parallel/CLAUDE.md)) for the parallel path.

> The dataset layer is where the design review's known-issues theme E lives (composition, mode flags,
> Hydra mechanics, and sharding all conflated in `data_organizer.py`/`dataset.py`) -- read
> `.agent/CLAUDE.md` section 2 and `espnet3_fable_review.md` before making non-trivial changes here.

## `modeling/`

- **`lightning_module.py`** -- `ESPnetLightningModule(lightning.LightningModule)`:
  `training_step`/`validation_step`, both the single-optimizer path (one `optimizer:`/`scheduler:`)
  and the multi-optimizer manual-optimization path (`optimizers:`/`schedulers:` keyed by name, e.g.
  for GAN-style training).
- **`optimization_spec.py`** -- `OptimizerSpec`, `SchedulerSpec`, `OptimizationStep`,
  `OptimizerRuntimeState`: typed config/state objects consumed by `ESPnetLightningModule`'s
  multi-optimizer path.

## `trainers/`

- **`trainer.py`** -- `ESPnet3LightningTrainer`: translates the `trainer:` config block
  (devices, strategy, logger, callbacks) into a `lightning.Trainer`, merging in
  `get_default_callbacks()`.

## `callbacks/`

- **`default_callbacks.py`** -- `AverageCheckpointsCallback` (implements
  `best_model_criterion`-driven checkpoint averaging), `MetricsLogger` (formats/logs per-epoch
  metrics), `get_default_callbacks()`.
- **`ema.py`** -- `EMACallback`: wraps `vendored_ema.EMA` as a Lightning callback (swaps EMA
  weights in for validation, restores online weights after).
- **`vendored_ema.py`** -- `EMA(Module)`: the underlying exponential-moving-average
  implementation, vendored from `lucidrains/ema-pytorch` (do not hand-edit without noting the
  upstream diff).

## `metrics/`

- **`base_metric.py`** -- `BaseMetric(ABC)`: the contract `systems/asr/metrics/*` implement
  (`__call__(hyp, ref, ...) -> dict`), including the shared SCP-reading (`iter_inputs`) helper.

## `optimizers/`

Placeholder package (docstring-only `__init__.py`). No custom optimizer classes exist yet; recipes
configure optimizers directly via Hydra (`optimizer: {_target_: torch.optim.Adam, ...}`) in
`training.yaml`.
