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

## System organization and naming

A System is the basic unit under `espnet3/systems/` and may represent a model,
a broader system, or a task when that is the clearest organization. Do not force
every System into the same taxonomy. Keep the external interfaces strict while
choosing the internal organization that gives the System a clear purpose.

Choose directory names in this order:

1. Use lowercase consistently.
2. Make the name clear and unambiguous.
3. Keep it as short as clarity allows.
4. Prefer one word when possible; avoid underscores when a clear single word
   works (for example, `f5tts`, `parakeet`, `granitespeech`, `speechlm`).
5. Use underscores when they express meaningful structure or distinguish
   variants (for example, `owsm_v4`, `qwen2_audio`, `qwen2_5_omni`). For
   ESPnet2-derived task systems, use `esp2_<task>` such as `esp2_asr` or
   `esp2_tts`. ESPnet2 provenance alone does not require the prefix; systems
   without a task-based structure, such as `speechlm`, use the ordinary naming
   rules.
6. Follow an established convention when it fits. Transformers' `models/`
   naming can inform the choice, alongside ESPnet3 clarity and maintenance.

Every System package provides `system.py` with its required System/stage
interface. Preserve the `metrics/` package structure for System metrics:
`metrics/__init__.py` and metric modules that follow the shared
[`BaseMetric` contract](../../../espnet3/components/metrics/base_metric.py),
including declared inputs/outputs and the required update/compute behavior.
Systems must also follow the shared inference contract; see the
[inference guide](../inference/AGENT.md). These interfaces are required even
when a System's internal taxonomy or module layout differs from other Systems.

## Adding or changing a stage

Before adding a stage, verify that it belongs in a System rather than a recipe
or reusable component. Update the concrete System, its configured stage list,
required-config validation, logging, and focused tests together. If a recipe
alone needs the behavior, prefer a recipe-local System; if multiple task
families need it, extend a shared abstraction.

Before adding a System, search for a clear existing name and apply the naming
rules above; discuss genuinely ambiguous names rather than maintaining an
exhaustive name catalog here.

## Verification

Run focused tests for the affected System and any shared stage contract. An
inference change may also affect the API, runner/provider, metrics, publication
bundle, or demo; trace those consumers and use the related package guides.
