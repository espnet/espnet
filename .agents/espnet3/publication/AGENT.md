# Publication and demos

See [shared contributor guidance](../../AGENT.md), the [inference guide](../inference/AGENT.md),
and [utilities](../utils/AGENT.md). Inspect current publication and demo code
before changing bundle or UI behavior.

Model publication packages configs, checkpoints, tokenizers, and other declared
artifacts into a portable bundle. Preserve relative-path rewriting and metadata
validation; the bundle must load outside the recipe checkout. When the bundle
schema changes, update its producer, versioning, loader, and tests together.

The supported consumer API is `espnet3.publication.InferenceModel`. Load a
local packed directory with `InferenceModel.from_packed(path)` or a Hub model
with `InferenceModel.from_pretrained("namespace/model")`, then call the result
with one sample; use `forward_batch(samples)` for batched inference. For example:

```python
from espnet3.publication import InferenceModel

model = InferenceModel.from_packed("/path/to/packed_model")
result = model({"speech": (16000, waveform)})
```

When a bundle contains recipe-local code, loading it requires
`trust_user_code=True`; only do this for code you trust. Do not add a separate
bundle loader or reimplement `InferenceModel`'s config loading, contract
conversion/validation, or backend invocation. Extend the shared API when it
needs a new capability so `infer`, packed models, and demos keep the same path.

Every publication change must verify that the produced pack actually runs via
`InferenceModel.from_packed`, using a representative sample and checking the
expected output. Run this from outside the recipe checkout when validating
portability; the publication integration check in the [CI guide](../../ci/AGENT.md)
is the reference. Also exercise `forward_batch` when the model supports batching.

ESPnet3 inference must use `InferenceRunner` or a compatible subclass in its
configured runner path. The shared runner/provider path supplies the same
inference behavior to recipe execution and `InferenceModel`, and the runner
provides the worker/shard execution used to parallelize inference. If a task
cannot use that runner contract, state the constraint in the ESPnet3 pull
request; without a compatible runner, it is outside the supported ESPnet3
inference and publication path.

Demo sessions load the same inference contract used by the infer stage. Keep UI
presentation and asset selection in the demo layer; model input/output names and
conversion remain defined by the inference contract. Repack after changing copied
configs or scripts, and verify field order and values through the session.

Declare inference input and output `contract` entries in the recipe's
`conf/inference.yaml`; this is the source of truth used by `InferenceModel` and
the demo. Check the packed `conf/inference.yaml` and load a `DemoSession` to
confirm the same fields, kinds, and conversions reach the UI and callback. If
`demo.yaml` supplies explicit UI input/output specs, keep them aligned with the
inference contract. Cover the full path from recipe inference through
`from_packed` to demo output in tests, following
[`test_shared_inference_paths.py`](../../../test/espnet3/publication/test_shared_inference_paths.py)
and the [publication integration check](../../../ci/test_integration_espnet3_publication_check.py).

Use focused tests for publication, loading, and demo changes. For portability,
run the relevant integration checks from the [CI guide](../../ci/AGENT.md).
Uploading to an external hub is a separate action requiring the user's request
and configured credentials.
