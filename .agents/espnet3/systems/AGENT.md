# `espnet3/systems/`

See [shared guidance](../../AGENT.md) and inspect
[`espnet3/systems/`](../../../espnet3/systems/) for the current System
implementations. This guide describes boundaries, not a fixed inventory of
systems, classes, or files.

## System and stage boundaries

A System owns the stages that its task family supports. `BaseSystem` provides
shared orchestration; concrete Systems define or override their own stages. The
dispatcher invokes requested stage methods, so a stage supported by one System
must not be presented as universal. In particular, tokenizer and data-filtering
stages are task-specific. Check the selected System class and its `run.py`
before documenting or invoking a stage.

Keep stage methods thin: read their stored stage configs, validate required
settings early, use shared logging and dispatch, and delegate the work to
components or helpers. Do not pass ad-hoc stage arguments through the
dispatcher.

Use `systems/base/` for behavior shared across task families. Keep model-facing
inference implementations in their task family and conform them to the public
[inference contract](../inference/AGENT.md). Do not put dataset sharding or output-writing
logic inside a model's inference class; those belong to the runner/provider and
writer contracts.

## Adding or changing a stage

Before adding a stage, verify that it belongs in a System rather than a recipe
or reusable component. Update the concrete System, its configured stage list,
required-config validation, logging, and focused tests together. If a recipe
alone needs the behavior, prefer a recipe-local System; if multiple task
families need it, extend a shared abstraction.

System directory names should be lowercase, clear, and consistent with nearby
systems. ESPnet2 task bridges use the established `esp2_<task>` pattern; for
other names, search the current tree and discuss ambiguous cases rather than
maintaining an exhaustive naming catalog here.

## Verification

Run focused tests for the affected System and any shared stage contract. An
inference change may also affect the API, runner/provider, metrics, publication
bundle, or demo; trace those consumers and use the related package guides.
