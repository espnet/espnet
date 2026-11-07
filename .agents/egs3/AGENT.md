# `egs3/` recipes

See [the shared contributor guide](../AGENT.md) for cross-cutting conventions
and [the systems guide](../espnet3/systems/AGENT.md) for System boundaries.
This guide keeps the recipe-creation workflow and the important contracts; for
current filenames and implementation details, inspect the referenced recipes
and source rather than maintaining a second exhaustive inventory here.

## Start from the right reference

For a new, full-scale ASR recipe using a standard System, use
[`egs3/librispeech_100/esp2_asr/`](../../egs3/librispeech_100/esp2_asr/) as the
primary working reference. It demonstrates a real corpus-specific `dataset/`,
self-contained configs, a thin `run.py`, and runnable stage commands. For a
small, fast CI fixture, inspect
[`egs3/mini_an4/esp2_asr/`](../../egs3/mini_an4/esp2_asr/); it is not a
substitute for a full-scale recipe. [`egs3/TEMPLATE/`](../../egs3/TEMPLATE/)
contains starter configs and shared run helpers, but is not a runnable recipe.
Check the current tree for a closer task-family or data-loading example when
LibriSpeech is not a good fit.

## Recipe structure and boundaries

A shipped recipe normally lives at `egs3/<corpus_name>/<system_name>/`. The
recipe owns corpus-specific preparation, dataset loading, configs, and local
helpers. A System owns reusable staged pipeline behavior. Use an existing
System whenever it provides the needed stages; add a recipe-local System only
for genuinely recipe-specific behavior. Put behavior shared by multiple
recipes in `espnet3/`, not in one recipe's `src/`.

Keep the recipe layout recognizable:

```text
egs3/<corpus_name>/<system_name>/
  conf/                 # training.yaml, inference.yaml, metrics.yaml, etc.
                        # variants use names such as training_*.yaml
  dataset/
    __init__.py
    builder.py          # source download/preparation and task-ready cache
    config.yaml         # corpus paths, URLs, environment variables, splits
    dataset.py          # recipe Dataset implementation
  src/
    __init__.py
    *.py                # recipe-specific, config-wired helpers
  run.py
  path.sh
  README.md
```

Use stage nouns for config filenames (`training.yaml`, `inference.yaml`,
`metrics.yaml`) and keep variants identifiable with a suffix. Keep both
`dataset/` and `src/` importable with their `__init__.py` files so recipe
datasets and config-wired helpers can be resolved by Python.

The LibriSpeech reference is a useful guide to how the pieces fit together:

- `dataset/` contains the corpus builder and `torch.utils.data.Dataset`.
- `conf/` contains stage configs such as `training.yaml`, `inference.yaml`,
  and `metrics.yaml` (plus publication/demo configs where used).
- `src/` contains recipe-specific helpers wired by config.
- `run.py` dispatches stages; `path.sh` sets up the checkout environment; and
  `README.md` documents data preparation and runnable commands.

Keep recipe configs self-contained: do not make one recipe's Hydra `defaults`
silently supply another recipe's settings. Use block-style YAML mappings, not
flow-style one-line mappings. Keep the README name exactly `README.md`.

## Creating a recipe

For a new shipped recipe, copy the closest working recipe (for standard ASR,
start with `egs3/librispeech_100/esp2_asr/`) into
`egs3/<corpus_name>/<system_name>/`, then adapt corpus-specific code and
settings. Inspect imports from `egs3.TEMPLATE` and `path.sh`: a copied recipe
may still depend on helpers in this checkout and is not automatically
standalone. Run the baseline/reference recipe first when feasible, so errors
introduced by the new data path are distinguishable from environment issues.

### Dataset implementation

Corpus-specific work usually belongs in `dataset/`. Its `__init__.py` must
expose the exact names `Dataset` and `DatasetBuilder` (they may alias
corpus-specific class names), because `BaseSystem.create_dataset()` resolves
those names. Follow the reference recipe's exports and the current
`DatasetBuilder` contract in
[`espnet3/components/data/dataset_builder.py`](../../espnet3/components/data/dataset_builder.py).
Recipe `Dataset` classes should inherit `BaseDataset` and declare the item
schema with `fields = (Field(...), ...)`. Every returned sample must include
the required declared fields with values matching their kinds; optional fields
must be marked optional in the declaration. See
[`espnet3/contract/data.py`](../../espnet3/contract/data.py) and the mini_an4
dataset for the working pattern. When adding or changing sample fields, update
the contract declaration and focused dataset/contract tests along with the
model, preprocessor, and metrics configs that consume them.

The builder separates raw-source availability from task-ready artifacts:

- `is_source_prepared` / `prepare_source` check for and obtain the raw corpus.
- `is_built` / `build` check for and create any manifests or converted data
  needed by the Dataset.

The `is_*` methods must be cheap, side-effect-free existence checks. Do not
download, create directories, or write caches in them. `prepare_source` should
download/extract data when appropriate, or clearly tell users how to obtain and
place data that cannot be redistributed. `build` should only create
task-ready artifacts. Keep these responsibilities separate; inspect the
LibriSpeech builder for a raw-layout passthrough and mini_an4 for a
manifest-building example. Prefer shared helpers in `espnet3.utils` for
downloads and extraction over ad-hoc network/archive code.

Training samples should contain only fields consumed by the model and
preprocessor. In inference datasets, expose scoring metadata such as `utt_id`,
`reco`, or reference labels only when the metrics config reads those columns via
`dataset:<column>`; keep them out of `input_key` and training batches unless the
model or preprocessor consumes them. Trace each field through the actual
training, inference, and metrics paths before adding it, because unsupported
training fields can break downstream consumers. Put corpus paths, split names,
and other builder-specific settings in `dataset/config.yaml` when that matches
the existing recipe pattern, and load them using the shared config helper
rather than hard-coding them in builder logic.

External dataset libraries (for example Lhotse or Hugging Face Datasets) can
be used behind a normal importable `Dataset` class selected through the
recipe's `data_src` config. Check the current dataset module-loading code and a
working recipe before adding framework support; a recipe-local adapter is
often sufficient.

### Configs, helpers, and stage dispatch

Start from the corresponding configs in the working reference or
`egs3/TEMPLATE/<system_name>/conf/`, then update the dataset source, tokenizer,
model, and recipe-specific settings. Check how the config is instantiated
before changing Hydra `_target_`, `_recursive_`, or interpolation behavior.
Keep each experiment understandable from its own YAML. Put tunable default
values in the appropriate TEMPLATE config and let recipes override them; do
not hide recipe or run defaults in module-level global variables.

Keep `run.py` as a thin dispatcher, following the selected System's working
recipe and the helpers under `egs3/TEMPLATE/`. Use the existing System class
and default stages whenever possible. If a genuinely recipe-specific stage is
needed, implement it in `src/system.py`, pass that System from `run.py`, and
follow the shared guidance for adding and testing stages. Do not add a wrapper
System just to pass config or select a model.

Helpers in `src/` should be added only when the recipe needs them and should
remain config-wired. Examples include custom inference-output construction,
training-text collection for a tokenizer, or a recipe-specific demo app. Check
the working recipe's call sites and the target interface before implementing
one; do not assume every recipe needs these hooks.

### README and validation

Document the actual data setup and copy-pasteable stage commands in the
recipe's `README.md`; use LibriSpeech's README as the quick-start pattern.
Commands should run from the documented directory, source `path.sh` where
needed, and prepare the dataset before stages that consume it. Keep `path.sh`
portable and avoid machine-specific paths or resource settings.

Add focused tests for non-trivial dataset preparation and recipe helpers. Run
the closest existing recipe test and relevant shared tests. Before committing
recipe changes, inspect YAML, Python, shell, and README files for hard-coded
local paths, credentials, and host-specific worker settings; resolve paths via
config/environment variables or documented overrides instead.
