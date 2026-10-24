# Publication and demos

See [shared contributor guidance](../../AGENT.md), the [inference guide](../inference/AGENT.md),
and [utilities](../utils/AGENT.md). Inspect current publication and demo code
before changing bundle or UI behavior.

Model publication packages configs, checkpoints, tokenizers, and other declared
artifacts into a portable bundle. Preserve relative-path rewriting and metadata
validation; the bundle must load outside the recipe checkout. When the bundle
schema changes, update its producer, versioning, loader, and tests together.

Demo sessions load the same inference contract used by the infer stage. Keep UI
presentation and asset selection in the demo layer; model input/output names and
conversion remain defined by the inference contract. Repack after changing copied
configs or scripts, and verify field order and values through the session.

Use focused tests for publication, loading, and demo changes. For portability,
run the relevant integration checks from the [CI guide](../../ci/AGENT.md).
Uploading to an external hub is a separate action requiring the user's request
and configured credentials.
