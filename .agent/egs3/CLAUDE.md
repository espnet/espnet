# `egs3/` -- recipes

See [`.agent/CLAUDE.md`](../CLAUDE.md) for cross-cutting guidance (dev setup, docstring/naming
conventions, known structural issues -- referenced below as "root guide").

```
egs3/
├── __init__.py
├── TEMPLATE/                  # NOT a runnable recipe -- the shared code/config every real recipe imports
│   ├── asr/
│   │   ├── run.py             # DEFAULT_STAGES, build_parser(), main() -- every ASR recipe's run.py imports these
│   │   ├── conf/{training,inference,metrics,publication,demo}.yaml   # fully-commented default configs
│   │   ├── src/{app.py, inference.py, hf_model_readme.md, hf_demo_readme.md}
│   │   └── readme.md
│   └── tts/
│       └── conf/{training,inference,metrics}.yaml   # no run.py / src/ yet -- no TTS recipe exists to date
├── mini_an4/asr/               # tiny CI recipe (an4 corpus); the one exercised by ci/test_integration_espnet3*.sh
│   ├── run.py                 # imports DEFAULT_STAGES / build_parser / main from egs3.TEMPLATE.asr.run
│   ├── conf/                  # 8 training-config variants + inference / inference_transducer / metrics / publication / demo
│   ├── dataset/{builder.py, dataset.py, config.yaml, __init__.py}  # resolved via `data_src: mini_an4/asr`
│   ├── src/{tokenizer.py, preprocessor.py, inference.py, app.py}
│   ├── path.sh, readme.md, downloads.tar.gz   # (prebuilt corpus fixture, so CI does not hit the network)
├── librispeech_100/asr/        # larger real-scale recipe, same shape as mini_an4 -- the reference recipe to
│   │                           # copy from when creating a new one (see below)
│   ├── run.py, conf/ (incl. conf/tuning/training_e_branchformer.yaml)
│   ├── dataset/{builder.py, dataset.py, config.yaml, __init__.py}
│   ├── src/{tokenizer.py, inference.py, app.py}
│   └── path.sh, readme.md
```

Cloning: `espnet3 clone <dataset>/<task> [--project DIR]` (see
[`espnet3/cli/CLAUDE.md`](../espnet3/cli/CLAUDE.md)) copies `conf/`, `dataset/`, `src/`, `run.py`,
`readme.md`, `path.sh` out of `egs3/` into a standalone directory that only needs `espnet3` installed
-- it no longer needs to live inside this checkout.

---

## Creating a new recipe

A recipe is "an all-in-one project under `egs3/`": configs, dataset code, recipe-local Python
helpers, a `run.py` entry point, and the output paths a system's stages write to. Use
**`egs3/librispeech_100/asr`** as the reference recipe to copy from -- it is the current example of a
real-scale recipe using a stock `System` and a real `dataset/` implementation (as opposed to
`egs3/mini_an4/asr`, which exists to be a tiny, fast fixture for CI, or `egs3/TEMPLATE/asr`, which is
not a runnable recipe at all).

### Two ways to start

- **Exploring / a personal project, outside this checkout**: `espnet3 clone librispeech/asr --project
  my_recipe`. This is the "clone a recipe and keep working in it" workflow: clone, run the baseline
  once, read through the copied `conf/` and `dataset/` code, then edit settings and dataset code in
  place and re-run `train`/`infer`/`measure` (or fine-tune) inside the same cloned project. Nothing
  under `my_recipe/` depends on the ESPnet checkout afterwards.
- **Contributing a new shipped recipe to this repo**: copy `egs3/librispeech_100/asr/` to
  `egs3/<dataset>/<task>/` by hand (not via `espnet3 clone`, which is meant for the first workflow)
  and edit the pieces below. Keep the same file layout so the shared `egs3.TEMPLATE.<task>.run` code
  and `espnet3 clone`/`--list` keep working for it.

### What to change, file by file

**`dataset/` -- almost always the part that actually differs between recipes.** This is where a new
corpus's specifics live; get this right before touching anything else.

- **`dataset/__init__.py`** must export exactly `Dataset` and `DatasetBuilder` names (aliased from
  your real class names), because `BaseSystem.create_dataset()` looks them up by those literal names
  (`DATASET_CLASS_NAME`/`DATASET_BUILDER_CLASS_NAME` -- root guide, Naming conventions). Copy
  librispeech_100's pattern:
  ```python
  from egs3.<dataset>.<task>.dataset.builder import MyBuilder as DatasetBuilder
  from egs3.<dataset>.<task>.dataset.dataset import MyDataset as Dataset
  __all__ = ["Dataset", "DatasetBuilder"]
  ```
- **`dataset/builder.py`** implements `espnet3.components.data.dataset_builder.DatasetBuilder`'s four
  methods: `is_source_prepared`/`prepare_source` (raw corpus availability -- download, copy, verify),
  `is_built`/`build` (task-ready artifacts derived from the source). Two real patterns to copy from:
  - **Raw-passthrough** (librispeech_100's `LibriSpeech100Builder`): the corpus is read directly from
    its original on-disk layout, so there is no separate build step -- `is_built`/`build` simply
    delegate to `is_source_prepared`/`prepare_source`. Use this when the upstream corpus format is
    already good enough for your `Dataset` class to read directly.
  - **Manifest-building** (mini_an4's `MiniAn4Builder`): `prepare_source` extracts a bundled/downloaded
    archive, `build` parses the raw transcripts and writes recipe-local manifests that `dataset.py`
    then reads. Use this when you need to normalize, filter, or re-key the source data before
    training can consume it.
  Keep `is_source_prepared`/`is_built` cheap and side-effect-free (they get called on every
  `create_dataset` run to decide whether to skip work) -- see the root guide's known-issues theme D
  for what goes wrong when a staleness check is too cheap (existence-only) or a build step is not
  atomic; do not repeat those mistakes in a new builder.
- **`dataset/dataset.py`** is the actual `torch.utils.data.Dataset`. `__getitem__` must return only
  the fields accepted by its model/preprocessor. Do **not** add `utt_id` to any recipe sample:
  ESPnet's dataset paths pass the full dictionary onward, where unsupported fields can break a stage.
  Recipe inference output must use its item index (or a framework-supported metadata mechanism) as
  its ID.
- **`dataset/config.yaml`** holds builder-specific settings that are not part of the training config
  (corpus sub-paths, an environment-variable name to check, the list of required splits). Load it
  once at import time with `espnet3.utils.config_utils.load_config_with_defaults`, exactly as both
  `librispeech_100` and `mini_an4` do -- do not hardcode these values inside `builder.py`.

**`conf/`** -- copy `training.yaml`/`inference.yaml`/`metrics.yaml`/`publication.yaml`/`demo.yaml`
from the reference recipe (or `egs3/TEMPLATE/<task>/conf/` for the fully-commented defaults) and
adjust: the `dataset:` block's `data_src`/`data_src_args` entries (or point them at your recipe's own
`dataset/` via a bare `data_src_args:` entry with no `data_src`, which resolves to the recipe-local
module -- see `espnet3/components/CLAUDE.md`'s `dataset_module.py` entry), the tokenizer settings, and
the model config. Keep `_recursive_: false` on the `dataset:` block (root guide, known-issues theme E
explains why it matters).

**`src/`** -- recipe-specific but framework-facing helpers, wired in by name from `conf/`:
- an inference-output builder such as librispeech_100's `build_output(data, model_output, idx)`,
  referenced from `inference.yaml` and called by `InferenceRunner` to shape what gets written to SCP;
- a tokenizer-text hook such as `gather_training_text(...)`, referenced from `training.yaml`'s
  `tokenizer.text_builder.func`;
- `app.py`, if the recipe ships its own demo UI beyond the shared `publication/demo` defaults.

**`run.py`** -- keep it a thin re-export, the same three lines every stock recipe has:
```python
from egs3.TEMPLATE.<task>.run import DEFAULT_STAGES, build_parser, main, parse_cli_and_stage_args
from espnet3.systems.<family>.system import <Family>System

if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, stages_to_run = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    main(args=args, system_cls=<Family>System, stages=stages_to_run)
```
Only replace `system_cls=<Family>System` with a custom system (below) or extend the stage list when
you actually need recipe-specific behaviour -- most recipes, including librispeech_100, use
`ASRSystem`/`TTSSystem` unmodified.

**`path.sh`, `readme.md`** -- environment setup (`PYTHONPATH`, `tools/activate_python.sh`) and a
quick-start section showing the real `--stages ...` invocations for this recipe, in the order they are
meant to be run (see librispeech_100's `readme.md` for the pattern: train -> infer -> measure, each as
its own copy-pasteable command).

### When you need a custom System

Use `ASRSystem`/`TTSSystem` directly whenever your recipe fits the standard stage set -- this covers
most recipes. Only subclass when you need genuinely recipe-specific behaviour that should not enter
the shared framework layer (`espnet3/systems/`): shared logic belongs in `espnet3/systems/`,
recipe-specific logic belongs inside the recipe.

Add the subclass at `egs3/<recipe>/<task>/src/system.py`:

```python
from espnet3.systems.asr.system import ASRSystem

class RecipeSystem(ASRSystem):
    def export_debug(self):
        output_dir = self.exp_dir / "debug_export"
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir
```

Then, in `run.py`: import `RecipeSystem` instead of `ASRSystem`/`TTSSystem`, add `"export_debug"` to
the stage list passed to `build_parser`/`main` (canonical order), and pass `system_cls=RecipeSystem`.
A stage is just a named method, so no separate registration step exists beyond what the root guide's
"Adding a new pipeline stage" section already covers -- follow that section for the full checklist
(config wiring, required-config validation, logging, tests) once the class itself exists.

### Checklist

- [ ] `dataset/__init__.py` exports `Dataset`/`DatasetBuilder` by those exact names
- [ ] `dataset/builder.py` implements all four `DatasetBuilder` methods; staleness checks are cheap,
      builds are idempotent (temp-file-then-rename if you write manifests -- known-issues theme D)
- [ ] `dataset/dataset.py` returns only fields accepted by the task/preprocessor; never add `utt_id`
- [ ] `conf/*.yaml` copied and adjusted (`data_src`, tokenizer, model, `_recursive_: false` kept)
- [ ] `run.py` is the thin three-line re-export unless a custom `System` is genuinely needed
- [ ] `readme.md` shows the real, working `--stages` commands for this recipe
- [ ] a unit test exists for any new/non-trivial code under `dataset/` or `src/`
      (mirrored at `test/espnet3/...` if it's promoted into shared `espnet3/` code later)
