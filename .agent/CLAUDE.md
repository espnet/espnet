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
| `egs3/` | [`egs3/CLAUDE.md`](egs3/CLAUDE.md) | Recipes (`TEMPLATE`, `mini_an4`, `librispeech_100`) + how to create a new one |
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

2. [Known structural issues](#2-known-structural-issues)
3. [Docstring guide](#3-docstring-guide)
4. [Naming conventions](#4-naming-conventions)
5. [Dev setup, linting, and PRs](#5-dev-setup-linting-and-prs)
6. [Adding a new pipeline stage](#6-adding-a-new-pipeline-stage)

(Creating a new *recipe*, as opposed to a new stage, is covered in `egs3/CLAUDE.md` instead, since it
is almost entirely about `egs3/` file layout.)

---

## 2. Known structural issues

A full read-only design and correctness review of `espnet3`/`egs3` was completed on this codebase
(commit `ec5632b167`). Results:

- **`espnet3_fable_review.md`** (English, repo root) / **`espnet3_fable_review.ja.md`** (Japanese) --
  the full report: executive summary, nine structural themes (A-I, report section 2b), every finding
  by submodule with file/line/scenario/fix, a test-quality audit of `test/espnet3`, and a prioritized
  punch list.
- **`review_log/BACKLOG.md`** -- the same findings as a flat fix-it checklist (206 items, every
  severity, one checkbox each). This is the list to work off of when actually fixing bugs.
- **`review_log/tools/recheck_after_merge.py`** -- run this after merging newer `upstream/master` to
  see which findings' cited lines have already changed (possibly already fixed) vs. still apply as-is.

The nine structural themes, in one line each (see report section 2b for the full write-up and the
recommended direction for each):

| # | Theme |
|---|---|
| A | The five per-recipe config files are used as a shared, mutable message bus between stages instead of an immutable resolved context. |
| B | The stage orchestrator is split across `systems/base/`, three files in `utils/`, and `egs3/TEMPLATE/asr/run.py` (library logic living inside a recipe). |
| C | No runtime rank/world-size model -- distributed-ness is inferred from config values, and non-train stages have no rank guard under Lightning's DDP launcher. |
| D | Idempotency/resume is implemented by directory existence everywhere; writes are non-atomic and shard locks can leak on failure. |
| E | The dataset layer (`DataOrganizer`/`CombinedDataset`) conflates composition, mode flags, Hydra mechanics, and sharding. |
| F | The `parallel` Runner/Provider abstraction leaks state (flat env namespace, `self`-capturing closures, per-call cluster lifecycle). |
| G | Trainer composition is hard-wired and checkpoint averaging/EMA correctness depends on Lightning's callback ordering. |
| H | `BaseSystem` is thin, so `asr`/`tts` diverge on constructor contract, `normalize` handling, and directory conventions. |
| I | The config surface is large with several documented-but-dead keys (no load-time validation). |

If you are about to touch `data_organizer.py`/`dataset.py` (theme E), `stages_utils.py`/`run_utils.py`
(themes A/B), or `parallel/` (theme F), read the corresponding report section first -- there is a good
chance the bug or refactor you are considering is already documented there with a concrete repro.

A follow-up review of documentation and docstrings (against section 3 below) is planned on a docs
branch; it will reuse the report's `docs-mismatch`-category findings as a starting point.

---

## 3. Docstring guide

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

**Private helpers** (`_`-prefixed, module-internal only) can stay short: a one-line summary, or a
short multi-line note explaining *why* a non-obvious branch exists. Do not force `Args:`/`Returns:`
sections onto something nobody outside the module will call directly.

**ESPnet3-specific things to call out explicitly, wherever they apply:**

- stage names (`train`, `infer`, `measure`, ...) a function/method participates in;
- which config file(s) a function reads from, and any cross-file field it expects `run_utils.py` to
  have already propagated (see theme A above -- if your function depends on a field being copied from
  another config, say so, because that dependency is not visible from the type signature);
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

## 4. Naming conventions

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

---

## 5. Dev setup, linting, and PRs

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
  (section 4 above). A new end-to-end feature needs integration coverage too -- see `ci/CLAUDE.md` for
  what "full workflow" (train -> infer -> measure -> publish -> demo) actually means here.
  If you are fixing something from `review_log/BACKLOG.md`, add a regression test that would have
  caught it -- several BACKLOG items exist precisely because the original test mocked past the bug
  (see the report's Test Quality section for named examples to avoid repeating).
- Request review from `@sw005320` and `@Masao-Someki` on espnet3 PRs.
- Common CI failure points: formatting (`black`/`isort`), docstring style (`flake8-docstrings`,
  section 3 above), shell script issues (`ci/test_flake8.sh`, shellcheck), and missing tests.

---

## 6. Adding a new pipeline stage

1. **Confirm it's really a new stage.** Check whether an existing stage covers it, whether a
   config-only change is enough, and whether the behaviour is recipe-specific (put it in the recipe's
   `src/`, not in shared `espnet3/`) before adding a shared stage.
2. **Implement it as a method on the relevant `System` subclass**, named with a short verb-style
   `snake_case` name (`prepare_labels`, `export_onnx`, ...). Shared logic goes in
   `espnet3/systems/<family>/system.py` (or a free function it calls, per section 4 above);
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
