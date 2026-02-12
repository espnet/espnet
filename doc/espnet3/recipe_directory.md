---
title: ESPnet3 Recipe Directory Layout
author:
  name: "Masao Someki"
date: 2025-11-26
---

## 🗂 ESPnet3 Recipe Directory Layout

ESPnet3 recipes live under `egs3/` and follow a consistent structure so you can
reuse tooling across corpora and tasks. This page explains the directory
layout, the role of `run.py`, and where to place configs and custom code.

## ✅ Where to put what

| Location                            | You put here                                      | Typical contents                       |
| ----------------------------------- | ------------------------------------------------- | -------------------------------------- |
| `egs3/<recipe>/<task>/conf/`       | YAML configs                                      | model, dataset, trainer, parallel     |
| `egs3/<recipe>/<task>/src/`        | Custom Python logic                               | dataset builders, scoring, extra utils|
| `egs3/<recipe>/<task>/run.py`      | Entry script wiring configs and systems           | stage definitions and CLI             |

## Directory structure

```
egs3/
  TEMPLATE/         # Minimal scaffold to copy for new recipes
    <system>/
      run.py        # Stage runner wiring
      readme.md     # Quickstart for the template
  <dataset>/  # Example corpus
    <system>/
      run.py        # Entry point (imports TEMPLATE logic)
      run.sh        # Optional helper scripts
      conf/         # Hydra/YAML configs and tuning variants
      src/          # Recipe-specific Python helpers
      readme.md     # Recipe-specific instructions
```

- **TEMPLATE** holds a working skeleton (Python stages, parser wiring). Copy it
  when starting a new recipe.
- **dataset/system** folders (e.g., `librispeech_100/asr`) customise configs and
  optional helpers but reuse the same stage runner.

## run.py: stage driver

`run.py` is the single entry point. It loads configs, instantiates a System
class, and executes the requested stages:

```python
from egs3.TEMPLATE.asr.run import ALL_STAGES, build_parser, main, parse_cli_and_stage_args
from espnet3.systems.asr.system import ASRSystem

if __name__ == "__main__":
    parser = build_parser(stages=ALL_STAGES)
    args, stages_to_run = parse_cli_and_stage_args(parser, stages=ALL_STAGES)
    main(args=args, system_cls=ASRSystem, stages=stages_to_run)
```

- **Stages** (`DEFAULT_STAGES`) define the lifecycle: `create_dataset`, `train_tokenizer`, `collect_stats`, `train`, `infer`, `measure`, `pack_model`, `upload_model`, `publish`.
- CLI flags select stages (`--stages train infer`) and configs
  (`--train_config conf/train.yaml`, `--infer_config conf/infer.yaml`, `--measure_config conf/measure.yaml`).

## Basic directory structure

- **conf/** stores YAML configs (train/eval, tuning variants).
- **data/** holds prepared datasets/manifests produced by stages.
- **exp/** is the experiment root for checkpoints, averaged models, and stats.
- **logs/** captures stdout/stderr per stage.
- **src/** is optional for recipe-specific Python helpers; import them from
  `run.py` or configs as needed.

## Creating a new recipe

1) Copy `egs3/TEMPLATE/<system>` into `egs3/<your_corpus>/<system>`.
2) Add configs under `conf/` (model, trainer, parallel, dataloader).
3) Point `run.py` to your `System` subclass (such as `ASRSystem` for training/evaluating ASR model).
4) Run stages with `python run.py --stages all --train_config conf/train.yaml`.

With this layout, every recipe shares the same stage driver while keeping data,
configs, and outputs contained per corpus/task.
