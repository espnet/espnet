# `egs3/` -- recipes

See [`.agent/CLAUDE.md`](../CLAUDE.md) for cross-cutting guidance (dev setup and
docstring/naming conventions).

```text
egs3/
├── TEMPLATE/                         # starter/reference files; not a runnable recipe
│   ├── esp2_asr/
│   │   ├── conf/{training,inference,metrics,publication,demo}.yaml
│   │   ├── run.py
│   │   └── src/
│   └── tts/
│       └── conf/{training,inference,metrics}.yaml
├── mini_an4/esp2_asr/                # small CI recipe
├── librispeech_100/esp2_asr/         # full ASR reference recipe
└── spgispeech/esp2_asr/              # full ASR recipe
```

## Recipe layout and system boundary

A shipped recipe always lives at **`egs3/<corpus_name>/<system_name>/`**.
`<corpus_name>` identifies the data source (for example, `librispeech_100` or
`mini_an4`); `<system_name>` identifies the implementation used to run it (for
example, `esp2_asr` or `tts`). Apply the system-directory naming rules in
[`.agent/espnet3/systems/CLAUDE.md`](../espnet3/systems/CLAUDE.md) to the second
component. Do not call it a `task` in paths, import names, or clone arguments.

A recipe combines a particular corpus with an instantiation of one System:
configs, dataset code, and any recipe-specific helpers. The System is the
framework-side staged pipeline. A recipe normally uses one existing System, but
may define a recipe-local System when it needs recipe-specific stages or when it
coordinates multiple systems, such as a multi-stage challenge recipe.

Every recipe directory has this structure:

```text
egs3/<corpus_name>/<system_name>/
├── conf/
│   ├── training.yaml                 # noun form of the train stage
│   ├── inference.yaml                # noun form of the infer stage
│   ├── metrics.yaml                  # noun form of the measure stage
│   ├── publication.yaml, demo.yaml   # noun form of their stages
│   └── <stage_noun>_*.yaml           # explicit variants, e.g. training_small.yaml
├── dataset/
│   ├── __init__.py
│   ├── builder.py                    # source download/preparation and cache creation
│   ├── config.yaml                   # dataset URLs, paths, environment variables, splits
│   └── dataset.py                    # torch.utils.data.Dataset implementation
├── src/
│   ├── __init__.py
│   └── *.py                          # recipe-specific, noun-named helpers
├── run.py
├── path.sh
└── README.md
```

`conf/` filenames use the noun form of the corresponding System stage method.
System code can enforce this config naming convention, but it does not prescribe
the rest of the recipe layout. `dataset/` has fixed core filenames. It may add
library-specific modules where useful, such as `lhotse_builder.py`,
`lhotse_dataset.py`, `hf_builder.py`, or `omniio_dataset.py`. `src/` remains
open to recipe-specific helpers, with ordinary noun-form filenames.

Keep every config self-consistent: one training experiment or one inference
trial should be understandable from its own YAML. Do not use Hydra `defaults`
to make one recipe config depend on another. Repetition that makes a config
self-contained is intentional.

Cloning uses the same two components: `espnet3 clone <corpus_name>/<system_name>
--project DIR`. It copies the concrete recipe into a standalone working project.

## Recipe README filenames

Every recipe README is named exactly `README.md`, with uppercase `README`.
Never add `readme.md`: filename casing matters on case-sensitive filesystems and
the conventional uppercase name keeps recipes consistent.

## TEMPLATE

`egs3/TEMPLATE/` provides starter/reference files and shared code where that code is
actually reused. It is not a runnable recipe and it must not create hidden config
dependencies. Copy a template config when starting a recipe, then keep the concrete
recipe config self-consistent.

---

## Creating a new recipe

A recipe is "an all-in-one project under `egs3/`": configs, dataset code, recipe-local Python
helpers, a `run.py` entry point, and the output paths a system's stages write to. Use
**`egs3/librispeech_100/esp2_asr`** as the reference recipe to copy from -- it is the current example of a
real-scale recipe using a stock `System` and a real `dataset/` implementation (as opposed to
`egs3/mini_an4/esp2_asr`, which exists to be a tiny, fast fixture for CI, or `egs3/TEMPLATE/esp2_asr`, which is
not a runnable recipe at all).

### Two ways to start

- **Exploring / a personal project, outside this checkout**: `espnet3 clone librispeech_100/esp2_asr --project
  my_recipe`. This is the "clone a recipe and keep working in it" workflow: clone, run the baseline
  once, read through the copied `conf/` and `dataset/` code, then edit settings and dataset code in
  place and re-run `train`/`infer`/`measure` (or fine-tune) inside the same cloned project. Nothing
  under `my_recipe/` depends on the ESPnet checkout afterwards.
- **Contributing a new shipped recipe to this repo**: copy `egs3/librispeech_100/esp2_asr/` to
  `egs3/<corpus_name>/<system_name>/` by hand (not via `espnet3 clone`, which is meant for the first workflow)
  and edit the pieces below. Keep the same file layout so the shared `egs3.TEMPLATE.<system_name>.run` code
  and `espnet3 clone`/`--list` keep working for it.

### What to change, file by file

**`dataset/` -- almost always the part that actually differs between recipes.** This is where a new
corpus's specifics live; get this right before touching anything else.

- **`dataset/__init__.py`** must export exactly `Dataset` and `DatasetBuilder` names (aliased from
  your real class names), because `BaseSystem.create_dataset()` looks them up by those literal names
  (`DATASET_CLASS_NAME`/`DATASET_BUILDER_CLASS_NAME` -- root guide, Naming conventions). Copy
  librispeech_100's pattern:
  ```python
  from egs3.<corpus_name>.<system_name>.dataset.builder import MyBuilder as DatasetBuilder
  from egs3.<corpus_name>.<system_name>.dataset.dataset import MyDataset as Dataset
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
  If a recipe needs to download or extract files, use the shared helpers in
  `espnet3.utils.download_utils` (`download_url`, `extract_targz`, and related utilities) instead of
  adding ad-hoc HTTP or archive handling. Test the download/extraction path with a small fixture or a
  local URL. The four methods have deliberately separate responsibilities:
  - `is_source_prepared` and `is_built` are cheap, side-effect-free predicates. They only inspect
    whether the raw source or task-ready cache exists and return a boolean. Do not download,
    extract, validate expensively, create directories, or write cache files from an `is_*` method.
  - `prepare_source` obtains the raw source. Download or extract it when the recipe can do so; when
    users must obtain the corpus themselves, emit a warning that explains the required source and
    where it must be placed.
  - `build` creates the task-ready cache, such as manifests or converted audio, and nothing else.
    Do not repeat an `is_built` check inside `build`: the caller has already made that decision via
    `is_built`. Keep cache writes atomic when a recipe generates artifacts.
  `is_source_prepared`/`is_built` get called on every `create_dataset` run to decide whether to skip
  work, so keep them fast and limited to their respective existence checks.
- **`dataset/dataset.py`** is the actual `torch.utils.data.Dataset`. `__getitem__` must return only
  the fields accepted by its model/preprocessor. Do **not** add `utt_id` to any recipe sample:
  ESPnet's dataset paths pass the full dictionary onward, where unsupported fields can break a stage.
  Recipe inference output must use its item index (or a framework-supported metadata mechanism) as
  its ID.
- **External dataset libraries** -- Lhotse, Hugging Face Datasets, and similar libraries do not need
  special framework support. Define a normal `Dataset` class in an importable module (and expose it
  as `Dataset`), then select that module from the recipe config with `data_src`. For example:
  ```yaml
  dataset:
    train:
      - data_src: my_project.datasets.lhotse_asr
        data_src_args:
          split: train
    valid:
      - data_src: my_project.datasets.hf_asr
        data_src_args:
          split: validation
  ```
  `data_src_args` is forwarded to that custom class. Supply a `DatasetBuilder` in the same module
  when the recipe also uses `create_dataset`.
- **`dataset/config.yaml`** holds builder-specific settings that are not part of the training config
  (corpus sub-paths, an environment-variable name to check, the list of required splits). Load it
  once at import time with `espnet3.utils.config_utils.load_config_with_defaults`, exactly as both
  `librispeech_100` and `mini_an4` do -- do not hardcode these values inside `builder.py`.

**`conf/`** -- copy `training.yaml`/`inference.yaml`/`metrics.yaml`/`publication.yaml`/`demo.yaml`
from the reference recipe (or `egs3/TEMPLATE/<system_name>/conf/` for the fully-commented defaults) and
adjust: the `dataset:` block's `data_src`/`data_src_args` entries (or point them at your recipe's own
`dataset/` via a bare `data_src_args:` entry with no `data_src`, which resolves to the recipe-local
module -- see `espnet3/components/CLAUDE.md`'s `dataset_module.py` entry), the tokenizer settings, and
the model config. Keep each config self-consistent. Keep `_recursive_: false` on the `dataset:` block when needed.

**`src/`** -- recipe-specific but framework-facing helpers, wired in by name from `conf/`:
- an inference-output builder such as librispeech_100's `build_output(data, model_output, idx)`,
  referenced from `inference.yaml` and called by `InferenceRunner` to shape what gets written to SCP;
- a tokenizer-text hook such as `gather_training_text(...)`, referenced from `training.yaml`'s
  `tokenizer.text_builder.func`;
- `app.py`, if the recipe ships its own demo UI beyond the shared `publication/demo` defaults.

**`run.py`** -- keep it a thin re-export, the same three lines every stock recipe has:
```python
from egs3.TEMPLATE.<system_name>.run import DEFAULT_STAGES, build_parser, main, parse_cli_and_stage_args
from espnet3.systems.<family>.system import <Family>System

if __name__ == "__main__":
    parser = build_parser(stages=DEFAULT_STAGES)
    args, stages_to_run = parse_cli_and_stage_args(parser, stages=DEFAULT_STAGES)
    main(args=args, system_cls=<Family>System, stages=stages_to_run)
```
Only replace `system_cls=<Family>System` with a custom system (below) or extend the stage list when
you actually need recipe-specific behaviour -- most recipes, including librispeech_100, use
`ASRSystem`/`TTSSystem` unmodified.

**System selection** -- use `BaseSystem` (or the applicable existing task-family System) when its
stages and behavior are sufficient. Do not introduce a wrapper class just to wire a recipe. When the
required behavior or stage is not supplied by the existing System, define a recipe-local System under
`src/system.py` and make `run.py` import and pass that class as `system_cls`.

**`path.sh`, `README.md`** -- environment setup (`PYTHONPATH`, `tools/activate_python.sh`) and a
quick-start section showing the real `--stages ...` invocations for this recipe, in the order they are
meant to be run (see librispeech_100's `README.md` for the pattern: train -> infer -> measure, each as
its own copy-pasteable command).

### When you need a custom System

Use `ASRSystem`/`TTSSystem` directly whenever your recipe fits the standard stage set -- this covers
most recipes. Only subclass when you need genuinely recipe-specific behaviour that should not enter
the shared framework layer (`espnet3/systems/`): shared logic belongs in `espnet3/systems/`,
recipe-specific logic belongs inside the recipe.

Add the subclass at `egs3/<corpus_name>/<system_name>/src/system.py`:

```python
from espnet3.systems.esp2_asr.system import ASRSystem

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

### Recipe portability review before push

Before pushing changes to a recipe, inspect its tracked YAML, Python, shell,
and README files for values that only work in the author's environment. Do not
commit absolute local paths, user-home paths, API keys, tokens, passwords, or
other credentials. Resolve dataset and output locations through recipe config,
environment variables, or documented command-line overrides instead.

Review `parallel` settings as well. Do not leave `n_workers: 1` or another
machine-specific resource value merely because it was needed on the development
host or in a local test. Keep a documented, generally useful default and make
site-specific worker counts an override.

When a user asks to push recipe changes, report this portability review and
call out any hard-coded path, credential, or environment-specific parallel
setting found. Never push a recipe containing credentials; remove them first.

- [ ] `dataset/__init__.py` exports `Dataset`/`DatasetBuilder` by those exact names
- [ ] `dataset/builder.py` implements all four `DatasetBuilder` methods; staleness checks are cheap
      and side-effect-free, `prepare_source` downloads or warns, and `build` only creates caches
      after `is_built` has decided they are needed (use temp-file-then-rename when writing manifests)
- [ ] `dataset/dataset.py` returns only fields accepted by the task/preprocessor; never add `utt_id`
- [ ] `conf/*.yaml` copied and adjusted (`data_src`, tokenizer, model, `_recursive_: false` kept)
- [ ] `run.py` is the thin three-line re-export unless a custom `System` is genuinely needed
- [ ] `README.md` shows the real, working `--stages` commands for this recipe
- [ ] a unit test exists for any new/non-trivial code under `dataset/` or `src/`
      (mirrored at `test/espnet3/...` if it's promoted into shared `espnet3/` code later)
