# Parallel execution

See [shared contributor guidance](../../AGENT.md). Inspect `espnet3/parallel/`
and the stage that uses it for the current runner/provider contract; do not keep
a second file or class inventory here.

Use the shared Runner/Provider pattern for work that can be split across
independent items. The runner owns planning, dispatch, shard persistence, and
merging. A provider owns the worker-local environment and processes its assigned
item. Keep scheduler-specific behavior out of recipes and model inference code.

Preserve resume and data-integrity behavior when changing shard plans or output
locations. Review manifests, locks, completion markers, and merge semantics;
reusing an output directory with a different item plan must fail clearly or be
handled by an explicit migration.

For model inference, follow the [inference contract](../inference/AGENT.md).
Model-level batching belongs in the model's batch hook; worker and shard
parallelism belongs in the runner/provider. Do not capture a runner instance in
a worker closure when the contract expects static worker functions and explicit
environment values.

Add focused unit tests for runner/provider behavior and a small integration
check when the stage is part of a recipe workflow. Use current implementations
and [CI guidance](../../ci/AGENT.md) to select the relevant checks.
