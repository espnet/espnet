# Inference

See [shared contributor guidance](../../AGENT.md) and the
[Systems guide](../systems/AGENT.md). There is no separate `espnet3.api`
package: the input/output declarations live in
[`espnet3/contract/`](../../../espnet3/contract/) (`inference.py`, `field.py`,
`kinds/`), the runner and provider in
[`espnet3/systems/base/`](../../../espnet3/systems/base/), and the packed-bundle
entry point `InferenceModel` in
[`espnet3/publication/inference_model.py`](../../../espnet3/publication/inference_model.py).
Read those and their tests; this guide records only the stable contract.

A recipe declares the `inputs` and `outputs` of inference in its inference
config using `Field` kinds. The shared contract validates and converts samples
before they reach the backend and validates the backend's outputs before they
are written. Keep model-specific logic in the backend the config instantiates,
and keep dataset iteration, sharding, and artifact writing in the runner and
provider.

For audio inputs, follow the declared channel and sample-rate contract. Do not
apply generic transposes or resampling in callers; use the kind's conversion.
Check the existing kinds before adding a new domain or modality. If inference
introduces an input or output value that the existing kinds cannot represent,
add a `Kind` under `espnet3/contract/kinds/` and register it in
`espnet3/contract/kinds/__init__.py`; declare the corresponding `Field` with
that kind. Define and test its validation and any required conversions for the
backend, demo, and streaming (`join`) paths rather than treating the new value
as an unvalidated generic object. Add focused contract tests and check the
runner, packed `InferenceModel`, and demo consumers.

`InferenceModel` reopens a packed bundle and rebuilds the backend from the
bundled inference config. Preserve validation of bundle paths, targets, and
trusted bundled code. The infer stage, `InferenceModel`, and the demo share the
same declarations, so check a change against all three consumers.

Use focused tests under `test/espnet3/contract/` and
`test/espnet3/publication/`, plus the affected system tests. Changes to fields
or call binding should also cover the runner and demo consumers; loading
changes should cover invalid configs and portable bundle paths. See the
[CI guide](../../ci/AGENT.md) for publication integration checks.
