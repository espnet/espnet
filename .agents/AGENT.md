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
  [API](espnet3/api/AGENT.md), [components](espnet3/components/AGENT.md),
  [parallel execution](espnet3/parallel/AGENT.md),
  [publication](espnet3/publication/AGENT.md),
  [systems](espnet3/systems/AGENT.md), or [utilities](espnet3/utils/AGENT.md).
- [CI guidance](ci/AGENT.md) for repository checks.

Use `rg --files` and inspect the relevant source tree to find current modules:
[`espnet3/`](../espnet3/), [`egs3/`](../egs3/), and [`ci/`](../ci/).
Do not duplicate a full source-tree listing here.

## Public APIs and docstrings

Use Google-style docstrings for public classes and functions. Every public
function or method docstring must include `Args`, `Returns`, and `Example`
sections, including when an argument list or return value is empty (`None` is
acceptable). Keep examples short and runnable. Describe purpose, input
constraints, output meaning, and meaningful failure cases. Public classes need
a concise purpose docstring and should document non-obvious fields or behavior.

Every private function or method (a name beginning with `_`) must have at least
a one-line docstring describing its purpose. Expand it when behavior or failure
cases are not obvious.

When changing a public contract, update its callers, tests, configs, and
consumers together. Validate configuration early and raise actionable errors.
Prefer an existing shared abstraction when it owns the behavior; put new shared
behavior in the appropriate `espnet3/` package rather than coupling task
families.

## Naming and terminology

Choose clear names that describe the operation or value. Function and method
names must start with a verb or verb phrase (for example, `load_graph`,
`validate_config`, or `is_ready`). Use nouns for classes, `PascalCase` for class
names, `snake_case` for functions, methods, and variables, and `UPPER_SNAKE_CASE`
for constants.

If a function, method, or class is only used inside its defining module, mark it
private with a leading underscore. Keep reusable public APIs public and avoid
exporting one-off implementation helpers. Prefer specific names over vague
names such as `handle`, `process`, or unexplained abbreviations. Follow nearby
domain terminology when it is established.

Use `infer`/`inference` for the end-to-end model-running stage and configuration.
Reserve `decode` for model-architecture components. Follow the local stage and
config terminology when adding a system or recipe.

Use `Base*` for ESPnet3 abstractions and preserve ESPnet2's `Abs*` names when
implementing its interfaces. For names whose convention is unclear, search
current definitions and call sites before introducing a new pattern.

## Configuration and pipeline conventions

Keep each recipe config understandable on its own. Use block-style YAML mappings
and inspect the actual Hydra/OmegaConf instantiation path before changing
`_target_`, recursion, interpolation, or conversion settings.

A System supports the stages implemented by that System. Do not assume that a
stage available in one task family is universal. Inspect the System class and
stage dispatcher before documenting, adding, or invoking a stage; unsupported
stages fail at dispatch.

Follow the existing public contract for dataset samples and inference outputs.
Do not add metadata fields to samples or outputs without checking every consumer,
including metrics, runners, publication bundles, and demos.

## Tests and changes

Place tests alongside the corresponding `test/espnet3/` or recipe test area.
Test observable behavior and failure cases; avoid changing a library interface
only to make a test easier to write. Run focused checks first, then any broader
CI command appropriate to the change.

Keep diffs focused. Do not commit credentials, machine-specific paths, or
unrelated formatting changes. See [CI guidance](ci/AGENT.md) for the commands
that match the repository's current checks.

`.agent/` is shared project guidance. Code ownership and review requests for
these files are declared in [`.github/CODEOWNERS`](../.github/CODEOWNERS);
ordinary branch protection and review rules determine whether approval is
required.
