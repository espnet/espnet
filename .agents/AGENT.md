# ESPnet3 contributor guide

This guide helps contributors and coding agents work in `espnet3/`, `egs3/`, and
related CI. It records stable conventions and points to the source for details;
the current code, tests, and shipped configs are authoritative. Avoid copying
directory trees, class inventories, or behavior descriptions that can drift.

Tools that support `AGENTS.md` should start at the repository-root
[`AGENTS.md`](../AGENTS.md). For this guide, follow the links below explicitly.

## Working procedure

1. Read the relevant package guide, implementation, callers, and tests.
2. Check `git status` and the diff; preserve existing work and stay in the
   assigned worktree.
3. Trace config targets, public contracts, stage dispatch, and downstream
   consumers before changing an interface.
4. Make the smallest change that fixes the issue, add or update focused tests,
   and update guidance only when a stable convention changed.
5. Run relevant checks from the [CI guide](ci/AGENT.md) and report what ran and
   any limitations.

## Where to look

- [Recipe guidance](egs3/AGENT.md) for `egs3/` work.
- [ESPnet3 package index](espnet3/AGENT.md), then the matching guide for
  [inference](espnet3/inference/AGENT.md), [components](espnet3/components/AGENT.md),
  [parallel execution](espnet3/parallel/AGENT.md),
  [publication](espnet3/publication/AGENT.md),
  [systems](espnet3/systems/AGENT.md), or [utilities](espnet3/utils/AGENT.md).
- [CI guidance](ci/AGENT.md) for repository checks.

Use `rg --files` and inspect the relevant source tree to find current modules:
[`espnet3/`](../espnet3/), [`egs3/`](../egs3/), and [`ci/`](../ci/).
Do not duplicate a full source-tree listing here.

## Public APIs and docstrings

Use concise Google-style docstrings for public classes and functions. Explain
purpose, inputs, outputs, and meaningful failure cases; include a runnable
example when configuration or calling conventions are not obvious. Keep private
helpers brief unless their behavior needs explanation.

When changing a public contract, update its callers, tests, configs, and
consumers together. Validate configuration early and raise actionable errors.
Prefer an existing shared abstraction when it owns the behavior; put new shared
behavior in the appropriate `espnet3/` package rather than coupling task
families.

## Naming and terminology

Choose clear names that match nearby code and describe the operation. Use verbs
that communicate whether a function constructs, resolves, validates, writes, or
runs something; follow a nearby implementation instead of maintaining a
repository-wide catalog of function names here.

Use `infer`/`inference` for the end-to-end model-running stage and configuration.
Reserve `decode` for model-architecture components. Follow the local stage and
config terminology when adding a system or recipe.

Use `Base*` for ESPnet3 abstractions and preserve ESPnet2's `Abs*` names when
implementing its interfaces. For names whose convention is unclear, search
current definitions and call sites before introducing a new pattern.

For new modules, match the filename pattern used by neighboring files in the
same directory. In particular, general-purpose helper modules under
`espnet3/utils/` use the `*_utils.py` suffix (for example, `config_utils.py` and
`logging_utils.py`); do not introduce a differently named utility module there
without a clear reason.

## Configuration and pipeline conventions

Keep each recipe config understandable on its own. Use block-style YAML mappings
and inspect the actual Hydra/OmegaConf instantiation path before changing
`_target_`, recursion, interpolation, or conversion settings.

Before introducing a module-level global, ask whether it is truly necessary.
Prefer function arguments, instance state, or explicit config for values that
vary by recipe, run, or caller. Use global state only when no clean alternative
fits, and make its process-wide lifetime and test/reset behavior clear. Defaults
that users may need to tune belong in the appropriate
`egs3/TEMPLATE/<system_name>/conf/` config, with recipe-specific overrides, not
in Python globals.

A System supports the stages implemented by that System. Do not assume that a
stage available in one task family is universal. Inspect the System class and
stage dispatcher before documenting, adding, or invoking a stage; unsupported
stages fail at dispatch.

Follow the existing public contract for dataset samples and inference outputs.
Do not add metadata fields to samples or outputs without checking every consumer,
including metrics, runners, publication bundles, and demos.

## Tests and changes

Place tests alongside the corresponding `test/espnet3/` or recipe test area.
Every new or changed unit test set must cover at least one normal case (valid
input gives the expected result) and one abnormal case (invalid input or a
documented failure raises the expected error with an actionable message). Test
observable behavior; avoid changing a library interface only to make a test
easier to write.

Before pushing, lint the changed files and run the pytest files that cover the
changed behavior (see [CI guidance](ci/AGENT.md) for the commands). Fix failures
rather than pushing them, and report any check you could not run. Then run any
broader CI command appropriate to the change.

Keep diffs focused. Do not commit credentials, machine-specific paths, or
unrelated formatting changes. See [CI guidance](ci/AGENT.md) for the commands
that match the repository's current checks.

`.agents/` is shared project guidance. Code ownership and review requests for
these files are declared in [`.github/CODEOWNERS`](../.github/CODEOWNERS);
ordinary branch protection and review rules determine whether approval is
required.
