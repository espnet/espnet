# ESPnet3 utilities

See [shared contributor guidance](../../AGENT.md). The utilities package holds
cross-cutting helpers for configuration, stage dispatch, logging, artifacts,
and publication. Inspect `espnet3/utils/` and its callers for the current module
layout rather than relying on a maintained file/function catalog.

When adding a general-purpose utility module here, follow the neighboring
filename convention and use the `*_utils.py` suffix (such as `config_utils.py`,
`logging_utils.py`, or `publication_utils.py`). Keep a different name only when
the module has a more specific established role or the existing directory
convention calls for it.

Keep task- or stage-specific behavior with its owner in `systems/` or
`components/`. Shared helpers should have clear inputs and side effects; resolve
configuration and validate required fields before a stage performs expensive
work. Preserve path handling for recipe-local configs and packed bundles.

When changing a helper, search its call sites and tests across `espnet3/`,
`egs3/`, and publication/loading paths. Add focused tests under
`test/espnet3/utils/` and run related checks from the [CI guide](../../ci/AGENT.md).
