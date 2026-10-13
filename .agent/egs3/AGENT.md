# `egs3/` recipes

See [the shared contributor guide](../AGENT.md) for cross-cutting conventions
and [the systems guide](../espnet3/systems/AGENT.md) for System boundaries.
The source tree is the inventory: do not maintain a second list of recipes or
systems here.

## Recipe structure

A recipe combines a corpus with a System and normally lives under
`egs3/<corpus>/<system>/`. Start from [`egs3/TEMPLATE/`](../../egs3/TEMPLATE/)
or a nearby working recipe whose data and task setup resemble the new one.
Templates are references, not runnable recipes. Inspect the current recipe tree
before copying; keep changes local to the recipe unless they implement reusable
framework behavior.

Keep a recipe's configs self-contained. Its `dataset/` code owns corpus-specific
preparation and sample loading; `src/` contains recipe-specific helpers. Export
the names expected by the dataset module loader and follow the current
`DatasetBuilder` contract. Keep source preparation separate from building
task-ready artifacts, and make existence checks cheap and side-effect-free.

Dataset samples should contain only fields accepted by the model and
preprocessor. Do not add utterance identifiers or other convenience metadata
without checking how the full sample flows through training, inference, and
metrics.

Use the shared System when its stages cover the recipe. Add a recipe-local
System only for genuinely recipe-specific behavior; put behavior needed by
multiple recipes or tasks in the shared framework. See the systems guide before
adding or assuming a stage.

## Configs and commands

Use the recipe's stage config files and keep each experiment's settings
understandable without implicit Hydra defaults from another recipe. Check the
actual instantiation path before changing `_target_`, `_recursive_`, or
interpolation behavior.

Use block-style YAML mappings. Keep `README.md` uppercase and give users
copy-pasteable commands that work from the documented directory, source the
recipe's `path.sh` where required, and include dataset creation before stages
that consume its outputs.

Keep `run.py` as a thin stage dispatcher. Do not add a wrapper System just to
pass config or select a model; make a custom System only when a real stage or
behavior requires it.

## Validation

Add focused tests for new dataset preparation, sample handling, and recipe
helpers. Run the closest existing recipe test and relevant shared tests. Before
committing or pushing, inspect YAML, Python, shell, and README changes for local
paths, credentials, and machine-specific resource settings.

For the current recipe layout and stage names, inspect `egs3/TEMPLATE/`, a
nearby working recipe, and the selected System implementation rather than
copying a static file or class catalog into this guide.
