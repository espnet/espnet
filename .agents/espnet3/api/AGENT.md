# Inference API

See [shared contributor guidance](../../AGENT.md) and the
[Systems guide](../systems/AGENT.md). Inspect `espnet3/api/inference/` for the
current implementation; this guide records its stable public contract.

An inference implementation declares its inputs and outputs with `Field`
objects and implements the `InferenceAPI` hooks. The API binds and validates
calls, converts declared values, and supports one-shot, streaming, and batched
execution where the model implements those hooks. Keep model-specific logic in
the system's inference implementation and keep dataset iteration, sharding,
and artifact writing in the system runner/provider.

For audio inputs, follow the declared channel and sample-rate contract. Do not
apply generic transposes or resampling in callers; use the API conversion and
the backend-specific wrapper where appropriate. Check the existing field kinds
before adding a new one.

Loading consumes a packed bundle's metadata and inference config to construct
the selected system implementation. Preserve validation of bundle paths,
targets, devices, and trusted bundled code. The infer stage and demo use the
same declarations, so changes must be checked across both consumers.

Use focused tests under `test/espnet3/api/` and the affected system tests.
Changes to fields or call binding should also cover runner/demo consumers;
loading changes should cover invalid schemas and portable bundle paths. See the
[CI guide](../../ci/AGENT.md) for publication integration checks.
