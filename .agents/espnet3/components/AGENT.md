# ESPnet3 components

See [shared contributor guidance](../../AGENT.md) and inspect `espnet3/components/`
for current modules. Components are reusable building blocks; stage orchestration
belongs in [Systems](../systems/AGENT.md), and work distribution belongs in
[parallel execution](../parallel/AGENT.md).

## Data

Data components organize configured datasets, instantiate recipe-provided
`Dataset`/`DatasetBuilder` implementations, and build the iterators consumed by
training and inference. Preserve the loader and iterator contracts when changing
batching or sharding. A recipe sample should contain only fields the configured
model and preprocessor consume; trace the sample through its full path before
adding metadata.

Recipe-specific source preparation and cache generation belong in the recipe's
dataset builder. Shared code should remain corpus-agnostic and expose any new
configuration behavior through a tested contract.

## Modeling and training

The Lightning module owns model steps and optimization behavior. Keep optimizer
and scheduler configuration consistent with the single- and multi-optimizer
paths, and validate incompatible combinations before training starts. Trainer
construction translates the `trainer:` config and combines user callbacks with
ESPnet defaults; do not assume a callback or stage is enabled unless the config
or implementation shows it.

Checkpoint selection, averaging, logging, and EMA are callback concerns. When
changing checkpoint or metric behavior, inspect both callback code and the
trainer that constructs it.

## Metrics and optimization

Metrics implement the shared metric contract and consume explicit input paths;
keep their names and input aliases aligned with metrics configs and inference
outputs. Prefer established external implementations for correctness-critical
metrics when available; otherwise follow the existing ESPnet implementation
and tests.

Optimizers and schedulers are normally selected by Hydra targets. Add a shared
implementation only when existing framework or PyTorch options cannot express
the behavior. Avoid maintaining an inventory of every class or module here;
the package source and tests are the index.

Add focused tests under `test/espnet3/components/` for changed contracts, config
validation, and distributed behavior. Use the relevant integration checks from
the [CI guide](../../ci/AGENT.md) when a component changes stage behavior.
