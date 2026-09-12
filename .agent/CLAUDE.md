# espnet3 developer guide -- index

Internal reference for anyone writing or reviewing code in `espnet3/`, `egs3/`, or `ci/`. It answers
"where does X live" and "what convention should I follow", not "how do I use espnet3 as an end user"
(for that, see each recipe's `readme.md` and the package docstrings themselves).

`.agent/` mirrors the source tree: every package that has its own reference doc keeps it at the
matching path (e.g. `espnet3/components/` is documented at `.agent/espnet3/components/CLAUDE.md`).
This file is the index plus the guidance that isn't specific to one package. All of these files are
hand-maintained; when a package's responsibilities change, update the matching doc in the same PR.

---

## 1. Where to look

| Path | Doc | Owns |
|---|---|---|
| `ci/` | [`ci/CLAUDE.md`](ci/CLAUDE.md) | CI scripts: lint, unit tests, integration tests, publication/demo smoke tests |
| `egs3/` | [`egs3/CLAUDE.md`](egs3/CLAUDE.md) | Recipes (`TEMPLATE`, `mini_an4`, `librispeech_100`, `aishell`) + how to create a new one |
| `espnet3/cli/` | [`espnet3/cli/CLAUDE.md`](espnet3/cli/CLAUDE.md) | The `espnet3` console script (`clone` subcommand) |
| `espnet3/systems/` | [`espnet3/systems/CLAUDE.md`](espnet3/systems/CLAUDE.md) | `BaseSystem`/`ASRSystem`/`TTSSystem` -- the staged pipeline |
| `espnet3/components/` | [`espnet3/components/CLAUDE.md`](espnet3/components/CLAUDE.md) | data / modeling / trainers / callbacks / metrics / optimizers building blocks |
| `espnet3/parallel/` | [`espnet3/parallel/CLAUDE.md`](espnet3/parallel/CLAUDE.md) | Runner/Provider execution model |
| `espnet3/publication/` | [`espnet3/publication/CLAUDE.md`](espnet3/publication/CLAUDE.md) | Model / demo packaging |
| `espnet3/utils/` | [`espnet3/utils/CLAUDE.md`](espnet3/utils/CLAUDE.md) | Cross-cutting helpers: config loading, stage dispatch, logging, ... |

There is also a top-level [`espnet3/CLAUDE.md`](espnet3/CLAUDE.md) with just the full `espnet3/`
directory tree and links to the subpackage docs above -- start there if you don't yet know which
subpackage you need.

This index also carries everything that applies across packages rather than to one of them:

2. [Docstring guide](#2-docstring-guide)
3. [Naming conventions](#3-naming-conventions)
4. [Dev setup, linting, and PRs](#4-dev-setup-linting-and-prs)
5. [Adding a new pipeline stage](#5-adding-a-new-pipeline-stage)

(Creating a new *recipe*, as opposed to a new stage, is covered in `egs3/CLAUDE.md` instead, since it
is almost entirely about `egs3/` file layout.)

---

## 2. Docstring guide

Docstrings are part of the public developer experience for both `espnet3/` and recipe code under
`egs3/`. Before writing one, make sure it answers: what does this do, when should I use it, what
inputs does it expect, what does it return or change, how can it fail, and (for anything non-trivial)
what does calling it actually look like.

Follow **Google-style** docstrings (this is what the rest of the codebase and `espnet2` already use):
a short one-line summary, a blank line, an optional longer description, then `Args:` / `Returns:` /
`Raises:` / `Notes:` / `Examples:` sections as needed.

**Public API** (anything importable from outside its own module -- classes, `System`/`Task` methods,
functions in `utils/`, `components/`, `parallel/`, `publication/`) should include:

- what the function/class actually does and when a caller should reach for it, in prose (not just a
  restatement of the signature);
- the config shape it expects, when it takes a `DictConfig`/`OmegaConf` object -- which keys it reads,
  which are optional, and what happens when an optional key is absent vs. `null`;
- `Args:`, `Returns:`, `Raises:` -- precise, not just types; say *why* an exception is raised, not just
  its class;
- `Examples:` when the calling convention is non-obvious (Hydra `_target_` wiring, a CLI invocation,
  multi-step usage) -- a runnable snippet or a realistic config fragment beats a vague description.
- **Conditional input contracts:** when a flag or config option changes an input's type, shape, or
  fields, document every supported branch and include a concrete example for each. For example,
  state both the flag value and the corresponding input shape rather than saying only that an option
  “changes the input format”.

**Private helpers** (`_`-prefixed, module-internal only) can stay short: a one-line summary, or a
short multi-line note explaining *why* a non-obvious branch exists. Do not force `Args:`/`Returns:`
sections onto something nobody outside the module will call directly.

**ESPnet3-specific things to call out explicitly, wherever they apply:**

- stage names (`train`, `infer`, `measure`, ...) a function/method participates in;
- which config file(s) a function reads from, and any cross-file field it expects `run_utils.py` to
  have already propagated;
- Hydra/OmegaConf-specific expectations: whether a `_target_`/`_recursive_`/`_convert_` value matters,
  and what breaks if it is missing (e.g. `DataOrganizer`'s `dataset:` block needs `_recursive_: false`
  in every shipped config -- omitting it changes when nested fields get instantiated);
  and `${...}` interpolations the caller is expected to have resolved already;
- dataset field naming a `Dataset`/`DatasetBuilder` implementation is expected to produce or
  consume; never add an unsupported `utt_id` field to a recipe sample because that dictionary is
  passed onward by the dataset pipeline;
- output directories a stage writes to, and whether re-running the stage is safe (idempotent) or not;
- whether the described behaviour lives in shared `espnet3/` code or is meant to be overridden/
  supplied by recipe-local code under `egs3/<recipe>/.../src/`.

**Avoid:**

- a docstring that only restates the function name (`"""Runs the run stage."""`);
- a long narrative of *how* the implementation works internally with no guidance on how to *call* it;
- an example that does not match how the function is actually invoked in a shipped recipe -- if you
  are not sure, grep `egs3/` for a real call site and base the example on that.

---

## 3. Naming conventions

**From the ESPnet3 contribution guide, verbatim:**

- New pipeline stages: short, verb-style `snake_case` method names on a `System` subclass (e.g.
  `prepare_labels`, `dump_features`, `export_onnx`), matched by a `--<stage>_config` CLI flag.
- Test files mirror source layout 1:1: a new `espnet3/foo/bar.py` gets tests at
  `test/espnet3/foo/test_bar.py` (this repo already follows that pattern -- keep it that way).

**Patterns already established in this codebase (not written down elsewhere, but follow them for
consistency when adding something new -- see the matching package doc from section 1 for the concrete
classes each pattern refers to):**

- **`*System`** -- the class owning a task family's staged pipeline (`BaseSystem`, `ASRSystem`,
  `TTSSystem`). One per task family, under `systems/<family>/system.py`.
- **`*Task`** -- a bridge to an `espnet2.tasks.abs_task.AbsTask` subclass (`ASRTask`,
  `ASRTransducerTask`), used only when a recipe sets `task:` instead of a direct Hydra `model:` target.
- **`*Runner`** / **`*Provider`** -- always appear in pairs: a `BaseRunner` subclass owns shard
  planning/dispatch, an `EnvironmentProvider` subclass owns "what one worker needs to process one
  item" (`InferenceRunner`/`InferenceProvider`, `CollectStatsRunner`/`CollectStatsInferenceProvider`,
  `RemoveLongShortRunner`/`RemoveLongShortProvider`). When adding a new parallel stage, name the pair
  the same way.
- **`*Callback`** -- a `lightning.Callback` subclass under `components/callbacks/`
  (`AverageCheckpointsCallback`, `EMACallback`).
- **`*Builder`** -- something that turns a config into a fully-constructed object with a multi-step
  contract: `DatasetBuilder` (per-recipe, in `dataset/builder.py`), `DataLoaderBuilder`.
  A recipe's dataset module must expose classes literally named `Dataset` and `DatasetBuilder`
  (`BaseSystem.DATASET_CLASS_NAME` / `DATASET_BUILDER_CLASS_NAME`) -- do not rename these in a recipe.
- **Free-function stage bodies**: `train`, `collect_stats`, `infer`, `measure`, `pack_model`,
  `upload_model`, `pack_demo`, `upload_demo` are implemented as plain functions taking a `DictConfig`
  (`systems/base/training.py`, `inference.py`, `metric.py`, `utils/publication_utils.py`,
  `publication/demo/packing.py`) and merely *called* by the matching `BaseSystem` method of the same
  name. Keep that split when adding a stage: the `System` method should stay a thin dispatcher.
  Private, module-internal step functions are prefixed with `_` and are not part of the public
  contract (e.g. `_build_trainer`, `_ensure_directories` in `training.py`).
  Module-level constants that gate default behaviour (`DEFAULT_STAGES`, `ALL_STAGES` in a recipe's
  `run.py`) are `UPPER_SNAKE_CASE`.
- Everything else follows standard PEP 8: modules and functions `snake_case`, classes `PascalCase`,
  constants `UPPER_SNAKE_CASE`.

**Naming requirements for new code:**

- File names, directory names, class names, and variable names must be nouns or noun phrases; function
  and method names must be verbs or verb phrases that describe the action they perform. This is about
  the *meaning* of the name, not its casing -- casing still follows PEP 8 (line above): file names,
  directory names, and variables stay `snake_case` nouns (`dataset_builder.py`, `recipe_dir`); only
  class names are noun phrases in `PascalCase` (`DatasetBuilder`). Do not write a file name in
  `PascalCase` or a class name in `snake_case`.
- Do not invent abbreviations. Use only the shortened forms that are already established across this
  codebase (`config`, `stats`, `exp`, `utils`, `dir`, `idx`, `hyp`/`ref`, `utt_id`, `env`, `spec`,
  `conf` inside espnet2-style model blocks). In particular write `config` / `*_config`, never `cfg` /
  `*_cfg` -- a handful of internal helpers still use `cfg`/`demo_cfg`/`writer_cfg`/`preprocessor_cfg`;
  that is drift to fix opportunistically, not a pattern to copy (section 3.7).
- When an existing function or component provides the same concept or behavior, use its established
  terminology and naming rather than introducing a synonym. Names for equivalent concepts must stay
  consistent across packages, configs, and recipes. In particular: this codebase's word for running a
  trained model on data is **`infer`/`inference`**, end to end -- the stage is `infer`, the config is
  `inference_config`/`inference.yaml`, the module is `inference.py`, the class is `InferenceRunner`/
  `InferenceProvider`/`InferenceModel`. Do not introduce `decode`/`decoding`/`decode_config` as a
  synonym for this, including in comments, readmes, or example paths -- `decode`/`decoder` is reserved
  for the model-architecture sense (an ASR decoder module, `decoder_conf`, `return_decoded_hyp`).
  Every shipped recipe readme currently labels its `--stages infer` step `# ... Decode` and one
  docstring uses `/exp/decode` as an example path; both are the drift this rule exists to stop, not
  something to match.

### 3.1 Verb vocabulary (function and method names)

The first word of a function name states what kind of thing it does. These are the verbs the codebase
actually uses, what each one means *here*, and the canonical examples to copy from.

| Verb | Meaning in this codebase | Canonical examples |
|---|---|---|
| `build_*` | Construct an in-memory object from config/components and **return it**; no persistent side effects. The dominant constructor verb. | `_build_trainer(config) -> ESPnet3LightningTrainer`, `build_client(config) -> Client`, `build_parser(stages) -> ArgumentParser`, `build_model`/`build_dataset`/`build_env_local`/`build_worker_setup_fn` (the `Provider` protocol), `build_output(data, model_output, idx)` (recipe `src/inference.py`), `build_input`/`build_output` (`UIAsset`), `_build_readme_context(...) -> dict` |
| `get_*` | **Look up or compute a small value that already exists** -- a path, a rank, a device, an existing registry entry. Returns cheaply, no construction of heavyweight objects. | `_get_lock_path(shard_dir) -> Path`, `_get_process_rank() -> int`, `get_parallel_config()`, `get_git_metadata(cwd)`, `get_class_path(obj) -> str`, `UIAssetRegistry.get(name)`, `_get_required_config(config, key, error_message)` |
| `resolve_*` | Turn a **reference, alias, interpolation, or relative path** into its concrete value -- a `Path`, a module name, a device string, a list of test-set names. Usually returns the concrete thing, may raise when it cannot be resolved. | `resolve_source_root(recipe_root, source_dir) -> Path`, `resolve_dataset_module_name(ref) -> str`, `resolve_stages(requested, stages) -> list[str]`, `_resolve_test_sets(metrics_config)`, `_resolve_device(config) -> str`, `_resolve_shard_dir(...) -> Path`, `resolve_loaded_configs(*configs)` (the one exception: resolves interpolations **in place**) |
| `load_*` | **Read and deserialize from disk** (or import a module) into an object: configs, manifests, dataset modules, packed bundles. | `load_config_with_defaults(path)`, `load_and_merge_config(...)`, `load_dataset_module(data_src, recipe_dir)`, `load_scp_paths(...)`, `_load_manifest()`, `load_demo_session(demo_dir, ...)`, `_load_builder_config()` (recipe pattern) |
| `on_*` | **Reserved for Lightning/Dask lifecycle hooks only** (`on_train_batch_end`, `on_validation_end`, `on_save_checkpoint`, ...). Never name an ordinary method `on_*`. | `EMACallback.on_*`, `AverageCheckpointsCallback.on_*`, `MetricsLogger.on_*` |
| `log_*` | Emit log records; returns `None`. Takes the `logging.Logger` as an explicit first argument when it is a module-level helper. | `log_run_metadata(logger, argv, configs, ...)`, `log_stage_metadata(logger, system, args)`, `log_component(logger, kind, label, obj)`, `log_dataloader(logger, loader, label)`, `_log_stats(mode, stats, weight)` |
| `write_*` | **Persist to a file**; returns the written `Path` when the caller needs it, else `None`. | `write_artifact(value, output_path, field_config) -> Path`, `write_record(writers, result, state, **env)` (the `Runner` protocol), `_write_manifest(shards) -> Path`, `_write_meta(...)`, `_write_bundle_config(...)` |
| `is_*` / `has_*` | Boolean predicates; **cheap, side-effect free**. `is_` for a state of the thing named, `has_` for possession of an optional part. | `is_source_prepared`, `is_built` (the `DatasetBuilder` protocol), `is_shard_done(shard_dir)`, `_is_tag(ref)`, `_has_tokenizer()`, `_has_exp_identity(config)`. Also `supports_integer_index`, `_uses_bundled_code`, `_matches_ignore_pattern` for predicates that read better as verbs. |
| `validate_*` / `validate` | Check a contract and **raise** a specific, actionable error when it fails; returns `None` (or the validated/normalized value, e.g. `_validate_written_path -> Path`). | `validate_experiment_context(...)`, `OptimizerSpec.validate()`, `_validate_strategy_compatibility(strategy)`, `_validate_output_with_keys(...)` |
| `collect_*` | Gather many items into one aggregate (statistics, keys, env vars). Also the stage name `collect_stats`. | `collect_stats(...)`, `collect_stats_batch(...)`, `_collect_string_keys(dataset)`, `_collect_env(prefixes, keys)` |
| `run_*` | **Execute** a stage, a shard, a subprocess, or an external tool; the thing that actually does the work. | `run_stages(system, stages_to_run, args, log)`, `_run_one_shard(shard_id, items, env)`, `_run_local(shards)` / `_run_parallel_dask(shards)`, `_run_git_command(cmd, cwd)`, `_run_pip_freeze()`, `_run_custom_writer(...)` |
| `prepare_*` | Make existing inputs ready for a later step **without performing that step**. | `prepare_source` (the `DatasetBuilder` protocol), `prepare_sentences(...)` (tokenizer input text), `_prepare_demo_config(...)`, `_prepare_training_runtime()` |
| `instantiate_*` | Specifically **Hydra `instantiate()` of a `_target_` config block**. Use this, not `build_`, when the object comes straight from `hydra.utils.instantiate`. | `_instantiate_model(config)`, `_instantiate_dataset(dataset_config, mode)`, `instantiate_dataset_reference(config, recipe_dir)`, `_instantiate_named_optimizers(specs)` |
| `ensure_*` | **Idempotent guarantee**: make a precondition true if it is not already (create a directory, import an optional dependency, inject a default key). No-op when already satisfied. | `_ensure_directories(config)`, `_ensure_jiwer()`, `_ensure_dask()`, `_ensure_target_convert_all(config)` |
| `iter_*` | Return an **iterator/generator**, never a materialized list. | `iter_inputs(data, *keys) -> Iterator`, `iter_source_candidates(...) -> Iterable[Path]`, `_iter_scp_file(file_obj)`, `_iter_attrs(obj)` (`_iter_outputs` returns a list -- do not copy that) |
| `infer_*` (+ stage `infer`) | Two meanings: the `infer` **stage** (`BaseSystem.infer`, `inference.infer(config)` -- see the terminology rule above: this and only this is the "run the model" verb, never `decode_*`), and **guessing a value from context** (`infer_artifact_type(value)`, `_infer_recipe_name(recipe_root)`, `_infer_system_name(...)`). Never name a new non-stage function plain `infer`. |
| `parse_*` | Interpret text / CLI args / a serialized entry into structured values. | `parse_cli_and_stage_args(parser, stages)`, `parse_dataset_reference_config(config)`, `_parse_transcript_line(line) -> tuple` |
| `normalize_*` | Canonicalize a value's **shape** (list vs scalar, path layout, sample dict) without changing its meaning. | `_normalize_key_list(keys)`, `_normalize_sample_for_runner(sample)`, `_normalize_downloads_layout(dataset_root)`, `_normalize_relative_resolver_paths(...)` |
| `copy_*` | File/tree copy, or copying values between config objects / between EMA and online weights. | `_copy_path(src, dst, ignore)`, `_copy_config_context(source, target, keys, ...)`, `copy_params_from_ema_to_model()` |
| `set_*` | Mutate **module-global or process-wide state** (logging format, the parallel context, a handler). Rare by design; every `set_` is a piece of global state you must account for. | `set_parallel(config)`, `set_stage_log_handler(log_dir, filename)`, `set_log_format(...)` |
| `create_*` | Only for **stage names** (`create_dataset`, `create_token_list`) and one factory (`create_inference_fn`). Not the general constructor verb -- that is `build_`. |
| `pack_*` / `upload_*` | Stage names only: `pack_model`/`upload_model`, `pack_demo`/`upload_demo`. |
| `register_*` | Add an entry to a registry / plugin table. | `UIAssetRegistry.register(name, asset, replace)`, `_register_worker_plugin(client, plugin, name)`, `_register_dataset_keys(...)` |
| `from_*` | `@classmethod` alternate constructors. | `OptimizerSpec.from_config(name, config)`, `InferenceModel.from_packed(pack_dir, ...)`, `InferenceModel.from_pretrained(model_tag, ...)` |
| `open_*` / `close_*` | The `Runner` writer lifecycle: `open_writers(shard_dir, **env) -> dict`, `close_writers(writers, state, **env)`. Paired; never one without the other. |
| `read_*` | Low-level line/row reading (`_read_lines_if_exists(path)`, `_read_manifest(path)`). Prefer `load_` for anything that returns a structured object. |
| `setup_*` / `setup` | Lifecycle hooks of an external framework (`EMACallback.setup(trainer, pl_module, stage)`, Dask `WorkerPlugin.setup(worker)`) plus `_setup_demo_assets`. Not a general "initialize" verb. |
| `apply_*` | Apply a transformation **in place** to config objects. | `apply_training_experiment_context(...)`, `_apply_substitutions(...)` |
| `train_*` (+ stage `train`) | The `train` stage and `train_tokenizer` / `train_sentencepiece` -- running a training procedure. |
| `forward*` | Model / provider inference entry points: `forward`, `forward_batch`, `forward_eval`. |
| `_step_*`, `_reset*`, `_swap_in_ema`, `_restore_online` | Optimizer/EMA state transitions inside `lightning_module.py` / `ema.py`; only meaningful in those classes. |

Verbs that appear **once** and should not be treated as conventions: `save_` (only `save_espnet_config`),
`dump_` (only `_dump_attrs`), `gather_` (only the recipe-local `gather_training_text`), `scan_`,
`persist_`, `materialize_`, `describe_`, `render_`, `expand_`, `chunk_`, `link_`. `make_` and
`execute_` do not occur at all -- do not introduce them; use `build_` and `run_` respectively.

**Choosing between neighbours:**

- `build_` vs `instantiate_` vs `get_`: `instantiate_` only when the result is a direct
  `hydra.utils.instantiate` of a `_target_`; `build_` for anything you assemble yourself; `get_` only
  when the value already exists and you are looking it up. `get_default_callbacks(...)` and
  `get_espnet_model(task, config)` construct things and are therefore misnamed by this rule -- do not
  copy them.
- `load_` vs `read_` vs `resolve_`: `load_` returns a structured object from disk; `read_` returns raw
  lines/rows; `resolve_` turns a *name or reference* into the concrete path/object **without** reading
  its contents.
- `write_` vs `save_`: the codebase uses `write_`. Do not add `save_*` for new files.
- `validate_` vs `ensure_` vs `is_`: `validate_` raises, `ensure_` fixes, `is_` reports.
- `run_` vs `create_` vs stage names: stage methods are the verb-noun stage name itself
  (`collect_stats`, `pack_model`); the free function that implements it has the **same** name in a
  module named after the stage; `run_` is for executing something on behalf of a caller (`run_stages`,
  `_run_one_shard`).

### 3.2 Canonical stage names

The stage vocabulary is fixed; reuse these exact names in `--stages`, `System` methods, config keys
and log labels: `create_dataset`, `train_tokenizer`, `collect_stats`, `train`, `infer`, `measure`,
`pack_model`, `upload_model`, `pack_demo`, `upload_demo`; TTS additionally `create_token_list`,
`remove_long_short`. New stages follow the same `verb` or `verb_noun` shape. Note the existing
mismatch between stage verbs and the modules implementing them (`train` -> `training.py`, `infer`
-> `inference.py`, `measure` -> `metric.py`); new stage modules should be named after the stage
itself.

### 3.3 Noun vocabulary (parameters, variables, config keys)

**Configs.** `config` is the DictConfig being operated on; the five recipe files are always
`training_config`, `inference_config`, `metrics_config`, `publication_config`, `demo_config`; sub-blocks
are `<block>_config` (`dataset_config`, `dataloader_config`, `model_config`). In YAML, top-level
espnet3 sections have no suffix (`dataset:`, `dataloader:`, `trainer:`, `parallel:`), while
espnet2-style nested model blocks keep espnet2's `*_conf` suffix (`normalize_conf`, `encoder_conf`) --
do not mix the two styles inside one block.

**Filesystem nouns.** Three suffixes with distinct meanings:

| Suffix | Meaning | Established names |
|---|---|---|
| `*_dir` | A directory the code **writes into or is configured with** | `recipe_dir`, `source_dir`, `output_dir`, `demo_dir`, `shard_dir`, `exp_dir`, `stats_dir`, `inference_dir`, `dataset_dir`, `data_dir`, `log_dir`, `out_dir` (legacy spelling; write `output_dir` for new code) |
| `*_root` | The **resolved top of a tree you read from**, derived from a `*_dir`/reference | `recipe_root`, `source_root`, `bundle_root`, `dataset_root`, `shards_root`, `egs3` root |
| `*_path` | A **file** path (or a path that may be file or dir when explicitly documented) | `output_path`, `config_path`, `manifest_path`, `transcript_path`, `demo_config_path`, `base_path`, `readme_template_path`, `archive_path`, `audio_path` |

`recipe_dir` is the *parameter* name (what the caller passes, from `training_config.recipe_dir`);
`recipe_root` is the *local variable* after `Path(recipe_dir).resolve()`. Keep that distinction.

**Data nouns.**

- `mode` -- which split a loader/stats pass is for: `"train"` / `"valid"` / `"test"` (the
  espnet3-level word). `split` -- the **corpus's own** split name inside a recipe `Dataset`
  (`"train-clean-100"`, `"dev-clean"`). Do not use one where the other is meant.
- `test_name` / `test_set` -- name of one inference/measure test set (the `name:` of a
  `dataset.test` entry); `test_sets` for the list.
- `idx` -- an integer index; `utt_id` -- utterance identifier string; `uid` -- the
  `CombinedDataset`-level string key. `idx_key`, `hyp_key`, `ref_key` -- the *names of the fields*
  holding index / hypothesis / reference in an output dict; `hyp`/`ref` -- the values.
- `data` -- the sample dict handed to `build_output(data, model_output, idx)`; `sample` -- a dataset
  item in `CombinedDataset`/`InferenceModel`; `batch` -- a collated batch inside Lightning steps;
  `model_output` -- the raw return of the model.
- `data_src` / `data_src_args` -- the dataset-reference config keys and their Python-side names.
- `shard`/`shards`, `shard_id` (identifier), `shard_idx` (position), `shard_dir`, `shard_subdir`,
  `shards_root` -- the `BaseRunner` sharding vocabulary; `items` -- the things being sharded;
  `manifest` -- the JSON bookkeeping file per shard root.
- `env` -- the worker environment dict a `Provider` builds and a `Runner` receives as `**env`;
  `state` -- the per-shard mutable dict passed through `open_writers` -> `write_record` ->
  `close_writers`; `writers` -- the dict `open_writers` returns; `params` -- provider constructor
  kwargs; `options` -- Dask cluster options.
- `spec`/`specs` -- a validated declarative description (`OptimizerSpec`, `SchedulerSpec`, UI
  `input_specs`/`output_specs`); `step` -- one `OptimizationStep`.
- `system` -- the `BaseSystem` instance passed to stage free functions (`pack_demo(system)`,
  `log_stage_metadata(logger, system, args)`); `system_cls` -- the class.
- `stages` -- the canonical ordered list; `stages_to_run` -- the resolved subset for this invocation;
  `stage` -- one name.
- `task` -- a dotted `espnet2.tasks.*` path string (`get_espnet_model(task, config)`) or the
  recipe-level task name (`asr`, `tts`); `task_path` when the dotted-path meaning must be explicit.
- `logger` vs `log`: both occur; **use `logger`** for `logging.Logger` parameters in new code.
- `name` -- a registry key or human label; `label` -- a display string for logs; `kind` -- a category
  string in `log_component`.

**Booleans / flags.** Predicate functions use `is_`/`has_`/`supports_`/`uses_`; boolean parameters
are bare adjectives or verbs: `train` ("is this the training split"), `resolve`, `strict`,
`replace`, `apply`, `write_collected_feats`, `trust_user_code`, `include_model_detail`. Avoid
`is_`-prefixed *parameters* (`is_train`) -- the codebase uses plain `train`.

**Counts.** `num_*` for counts in config and code (`num_device`, `num_nodes`, `num_items`,
`n_workers` is the one established exception because it mirrors Dask's own argument name).

### 3.4 Class names

| Suffix / prefix | Meaning | Members |
|---|---|---|
| `Base*` | Abstract base class **defined in espnet3** (espnet2 uses `Abs*` -- keep `Base*` for espnet3, `Abs*` only when subclassing espnet2's) | `BaseSystem`, `BaseRunner`, `BaseMetric` |
| `*System` | Owns one task family's staged pipeline | `ASRSystem`, `TTSSystem` (+ recipe-local `RecipeSystem` per `egs3/CLAUDE.md`) |
| `*Task` | espnet2 `AbsTask` bridge | `ASRTask`, `ASRTransducerTask` |
| `*Runner` / `*Provider` | Always a pair: shard dispatch / per-worker environment | `InferenceRunner`/`InferenceProvider`, `CollectStatsRunner`/`CollectStatsInferenceProvider`, `RemoveLongShortRunner`/`RemoveLongShortProvider`, base `EnvironmentProvider` |
| `*Builder` | Multi-step constructor with a protocol | `DatasetBuilder` (+ recipe `MiniAn4Builder`, `LibriSpeech100Builder`, `AishellBuilder`), `DataLoaderBuilder` |
| `*Dataset` | A `torch.utils.data.Dataset` | `CombinedDataset`, `ShardedDataset`, `DatasetWithTransform`, recipe `MiniAn4Dataset`/`LibriSpeech100Dataset`/`AishellDataset` |
| `*Example` / `*Entry` | An immutable **row record** inside a recipe dataset (dataclass) | `LibriSpeechExample`, `AishellExample`, `ManifestEntry` (pick `*Example` for new recipes; `ManifestEntry` is the outlier) |
| `*Callback` | `lightning.Callback` subclass | `AverageCheckpointsCallback`, `EMACallback` |
| `*Spec` / `*Step` / `*State` | Declarative config object / one unit of work / mutable runtime state (all in `optimization_spec.py`) | `OptimizerSpec`, `SchedulerSpec`, `OptimizationStep`, `OptimizerRuntimeState` |
| `*Config` | A typed view over a config block | `DatasetConfig` |
| `*Metric` / bare acronym | Metric implementations subclass `BaseMetric` but are named by the metric itself | `CER`, `WER`, `TER` |
| `*Module` / `*Trainer` | Lightning integration | `ESPnetLightningModule`, `ESPnet3LightningTrainer` (note the inconsistent `ESPnet`/`ESPnet3` prefix -- use `ESPnet3*` for new Lightning-facing classes) |
| `*Iterator` | Custom iterator | `EpochSyncIterator` |
| `*Model` | A loadable, callable inference wrapper | `InferenceModel` |
| `*Session` | Runtime object living for one demo/app process | `DemoSession` |
| `*UI` / `*Asset` / `*Registry` | Gradio UI building blocks and their registry; `Default*` for the built-in fallbacks | `UIAsset`, `DefaultAudioUI`, `DefaultTextUI`, `UIAssetRegistry` |
| `*Plugin` | Dask worker plugin | `DictReturnWorkerPlugin` |
| `*Preprocessor` | espnet2 `AbsPreprocessor` subclass in a recipe | `MiniAn4TokenizeSpeedPerturbPreprocessor` |
| `*Progress` | Progress reporter | `DownloadProgress` |

Recipe class names are `<Corpus><Kind>`: `MiniAn4Builder`, `LibriSpeech100Dataset`,
`AishellExample`; the corpus part is CamelCased from the recipe directory name and kept identical
across `Builder`/`Dataset`/`Example`. The recipe's `dataset/__init__.py` then re-exports them under
the fixed names `Dataset` and `DatasetBuilder`.

### 3.5 Methods that belong to a protocol

These method names are **contracts** read by other code; implement them with exactly these names and
signatures, and do not reuse the names for unrelated methods:

- `DatasetBuilder`: `is_source_prepared(**kwargs) -> bool`, `prepare_source(**kwargs) -> None`,
  `is_built(**kwargs) -> bool`, `build(**kwargs) -> None`. Recipe implementations take
  `recipe_dir, source_dir=None, **_kwargs` (the leading-underscore `**_kwargs` marks
  "accepted for API compatibility, unused").
- `EnvironmentProvider` / `Provider`s: `build_env_local() -> dict`, `build_worker_setup_fn() ->
  Callable`, plus `build_model(config)`, `build_dataset(config)` on inference providers.
- `BaseRunner` / `Runner`s: `open_writers(shard_dir, **env) -> dict`, `write_record(writers, result,
  state, **env) -> None`, `close_writers(writers, state, **env)`, `is_shard_done(shard_dir) -> bool`,
  `shard_task(...)`.
- `BaseMetric`: `__call__(...) -> dict`, `iter_inputs(data, *keys) -> Iterator`.
- espnet2 `AbsTask` (via `ASRTask`): `build_model(args)`, `build_collate_fn(args, train)`,
  `build_preprocess_fn(args, train)`, `add_task_arguments(parser)`, `required_data_names`,
  `optional_data_names`.
- Recipe `src/inference.py`: `build_output(data, model_output, idx) -> dict`; recipe
  `src/tokenizer.py`: a text builder such as `gather_training_text(...) -> list[str]` referenced from
  `tokenizer.text_builder.func`; recipe `src/app.py`: `build_demo(demo_dir, demo_config_path)`.
- Lightning: `training_step`, `validation_step`, `configure_optimizers`, `train_dataloader`,
  `val_dataloader`, `on_*`, `state_dict`/`load_state_dict`, `setup(trainer, pl_module, stage)`.

### 3.6 Modules, constants, and visibility

- **Module names** are nouns: `<noun>_utils.py` under `utils/` (`config_utils`, `run_utils`,
  `stages_utils`, `task_utils`, `logging_utils`, `publication_utils`, `writer_utils`, `scp_utils`,
  `download_utils`); `base_<noun>.py` for an ABC (`base_runner.py`, `base_metric.py`);
  `<noun>_provider.py` / `<noun>_runner.py` for a Runner/Provider pair; `default_<noun>s.py` for
  built-in defaults (`default_callbacks.py`); `vendored_<lib>.py` for copied third-party code;
  `<noun>_spec.py` for declarative spec dataclasses. Recipes use fixed file names: `dataset/{builder,
  dataset}.py` + `config.yaml`, `src/{inference,tokenizer,preprocessor,app}.py`,
  `conf/{training,inference,metrics,publication,demo}.yaml`, `conf/tuning/<model_variant>.yaml`.
- **Module-level constants** are `UPPER_SNAKE_CASE`; private ones carry a leading underscore
  (`_RANK_ENV_KEYS`, `_TRAINING_CONTEXT_KEYS`, `_DATA_SRC_KEY`, `_LOGGED_ENV`). Established suffixes:
  `*_RE` for a compiled regex (`TRANSCRIPT_RE`, `_FRONT_MATTER_RE`), `*_KEYS` for a tuple of config
  keys, `*_MAP` for a lookup dict (`CLUSTER_MAP`), `*_FORMAT` for format strings (`LOG_FORMAT`,
  `DATE_FORMAT`). Recipe modules load their `config.yaml` once at import time into `_CFG` /
  `_BUILDER_CFG` / `_DATASET_CFG` and derive `_KNOWN_SPLITS`/`_CONFIG_RESOURCE` from it -- reuse those
  exact names in a new recipe.
- **Visibility**: a leading underscore means module-internal; `**_kwargs` means "accepted, ignored".
  Anything without an underscore in `espnet3/` is public API and needs a full docstring (section 2)
  and a mirrored test.

### 3.7 Inconsistencies already in the tree (do not propagate)

Known drift, listed so a reviewer can say "use the established form" with a reference:
`cfg`/`demo_cfg`/`writer_cfg`/`preprocessor_cfg` (use `config`/`*_config`); `out_dir` (use
`output_dir`); `log` for a `Logger` (use `logger`); `get_default_callbacks`/`get_espnet_model`
(constructors named `get_`); `_iter_outputs` returning a list; `save_espnet_config` (lone `save_`);
`ManifestEntry` vs `*Example`; `ESPnetLightningModule` vs `ESPnet3LightningTrainer`; stage `measure`
implemented in `metric.py`; two different classes both named `InferenceProvider`
(`parallel/inference_provider.py` and `systems/base/inference_provider.py`). These are naming cues,
not permission to rename an existing public API without a compatibility plan.

---

## 4. Dev setup, linting, and PRs

**Environment** (Pixi + uv):

```bash
curl -fsSL https://pixi.sh/install.sh | bash
git clone https://github.com/espnet/espnet.git && cd espnet
pixi init && pixi add python=3.11 pip ffmpeg
pixi shell               # or: eval "$(pixi shell-hook)"
uv pip install -e .
uv pip install -e ".[asr]"    # add extras as needed, e.g. also "[tts]" or "[enh]"
```

**Before opening a PR, run locally** (mirrors `ci/test_python_espnet3.sh`, see `ci/CLAUDE.md`):

First verify that the active environment provides `black`, `isort`, `pycodestyle`, `flake8`, and
`flake8-docstrings`. Install any missing package before running the checks, for example:

```bash
python -m pip install black isort pycodestyle flake8 flake8-docstrings
```

Run the following checks before every push. The docstring check is particularly important: new code
often fails CI on `flake8-docstrings` even when formatting and ordinary lint checks pass.

```bash
black espnet3/ test/espnet3/ ci/
isort espnet3/ test/espnet3/ ci/
pycodestyle espnet3/ test/espnet3/ ci/
bash ci/test_flake8.sh espnet3
pytest -q test/espnet3/                 # or a smaller scope, e.g. pytest -q test/espnet3/systems/asr/
```

**PR expectations:**

- Keep PRs small: roughly 20 changed files / 2000 changed lines as a soft ceiling.
- Every change to `espnet3/` needs a new or updated unit test at the mirrored `test/espnet3/` path
  (section 3 above). A new end-to-end feature needs integration coverage too -- see `ci/CLAUDE.md` for
  what "full workflow" (train -> infer -> measure -> publish -> demo) actually means here.
- Every new `System` also needs an integration test. Use `mini_an4` and a deliberately small model
  configuration to keep it fast, but exercise both model construction and the recipe's `run.py`
  stage dispatch. Extend `ci/test_integration_espnet3.sh` when the standard pipeline is sufficient;
  add a focused integration script only when the new system cannot fit that harness.
- Request review from `@sw005320` and `@Masao-Someki` on espnet3 PRs.
- Common CI failure points: formatting (`black`/`isort`), docstring style (`flake8-docstrings`,
  section 2 above), shell script issues (`ci/test_flake8.sh`, shellcheck), and missing tests.

---

## 5. Adding a new pipeline stage

1. **Confirm it's really a new stage.** Check whether an existing stage covers it, whether a
   config-only change is enough, and whether the behaviour is recipe-specific (put it in the recipe's
   `src/`, not in shared `espnet3/`) before adding a shared stage.
2. **Implement it as a method on the relevant `System` subclass**, named with a short verb-style
   `snake_case` name (`prepare_labels`, `export_onnx`, ...). Shared logic goes in
   `espnet3/systems/<family>/system.py` (or a free function it calls, per section 3 above);
   recipe-only logic goes in `egs3/<recipe>/<task>/src/`. Keep the method itself thin -- move any
   nontrivial logic into its own module.
3. **Register it in `run.py`.** Add the stage name to the canonical `ALL_STAGES` list (or
   `DEFAULT_STAGES` if it should run by default). **Stage execution order always follows this list's
   order**, regardless of the order the user passes to `--stages`.
4. **Wire configuration through a `--<stage>_config` flag**, not ad-hoc CLI arguments -- load it with
   `utils/config_utils.load_and_merge_config` and store it as an instance attribute on the `System`,
   the same way `training_config`/`inference_config`/... are stored today.
5. **Validate required configs up front** in `run.py` (see the existing `required_configs` /
   missing-config check pattern) so a misconfigured recipe fails immediately with a clear message
   instead of partway through the stage.
6. **Use the shared logging utilities**: `configure_logging()` once, `log_stage`/`log_stage_metadata`
   around the new stage, and add it to `stage_log_mapping` in the `System`'s `__init__` if it needs a
   non-default log directory. Let `run_stages()` (`utils/stages_utils.py`) drive dispatch rather than
   calling the method directly from a new code path.
7. **Add tests**: a unit test for the new method/module at its mirrored `test/espnet3/...` path, a
   runner-level test if you touched `run.py`'s stage dispatch, and an integration-test addition if the
   stage is meant to run end-to-end in CI (extend `ci/test_integration_espnet3.sh` or add a new CI
   script following its pattern -- see `ci/CLAUDE.md`).
8. **Update docs**: the recipe's `readme.md` if the stage is recipe-specific, and the matching package
   doc under `.agent/` (section 1) if it changes what a shared package owns.
