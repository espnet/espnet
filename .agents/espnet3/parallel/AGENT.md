# Parallel execution

See [shared contributor guidance](../../AGENT.md). Inspect `espnet3/parallel/`
and the stage that uses it for the current runner/provider contract; do not keep
a second file or class inventory here.

Use the shared Runner/Provider pattern for work that can be split across
independent items. The runner owns planning, dispatch, shard persistence, and
merging. A provider owns the worker-local environment and processes its assigned
item. Keep scheduler-specific behavior out of recipes and model inference code.

Before implementing a runner/provider, use the existing code as a design
reference: [`InferenceRunner`](../../../espnet3/systems/base/inference_runner.py)
and [`InferenceProvider`](../../../espnet3/systems/base/inference_provider.py)
show the model-inference path, with their construction in
[`base/inference.py`](../../../espnet3/systems/base/inference.py). The
[`F5TTS remove_long_short stage`](../../../espnet3/systems/f5tts/src/remove_long_short.py)
is a compact example of a task-specific runner/provider pair that processes
manifest entries without loading a model. Follow the example closest to the
work, including how it builds worker-local state and returns results, instead
of duplicating dispatch or shard management in the stage.

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
