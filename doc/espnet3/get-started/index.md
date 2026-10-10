---
title: Getting Started with ESPnet3
author:
  name: "Masao Someki"
date: 2025-11-26
---

# 🚀 Getting Started with ESPnet3

This guide provides the fastest way to start using ESPnet3.
Choose the workflow that fits your environment and follow the examples below.


# ⚡ Quick Start (ASR Example)

## 1. Install ESPnet3

Pick a system that is closest to your work.
And then you can install only packages that is required for your work.

<HomeQuickStart />

## 2. Clone a recipe

Pick a built-in recipe and clone it into a local project directory.
The clone command copies the full recipe, so you can run it immediately without touching the ESPnet3 source tree.

```bash
espnet3 clone librispeech/asr --project my_project
cd my_project
python run.py
```

The cloned project looks like this:

```
my_project/
├── run.py                  # entry point — run stages from here
├── path.sh                 # PYTHONPATH / environment setup, sourced by shell scripts
├── readme.md                # quick-start commands for this recipe
├── conf/
│   ├── training.yaml        # model, trainer, optimizer, dataloader
│   ├── inference.yaml       # decoder settings, test sets
│   ├── metrics.yaml         # measure-stage metrics
│   ├── publication.yaml     # pack_model / upload_model
│   └── demo.yaml            # pack_demo / upload_demo
├── dataset/
│   ├── __init__.py          # exports `Dataset` and `DatasetBuilder`
│   ├── builder.py           # DatasetBuilder used by create_dataset
│   ├── dataset.py            # the torch Dataset class
│   └── config.yaml           # builder-specific settings (paths, env vars)
└── src/
    ├── tokenizer.py           # tokenizer-text hook wired from training.yaml
    ├── inference.py            # build_output(), wired from inference.yaml
    └── app.py                  # demo UI, if the recipe ships one
```

To see how each file in the cloned project connects to a stage, see
[How files and stages connect](./what-is-a-recipe.html#how-files-and-stages-connect).

See [Our Github recipes](https://github.com/espnet/espnet/tree/master/egs3) for the full list of recipes.

::: note
The cloned directory is not just a sample.
It is your working project for baseline runs, fine-tuning, inference, and
publication.
:::

## 3. Run with run.py

Each stage maps to a named step in the pipeline. Run them one at a time or all at once.

```bash
# Step by step
python run.py --stages create_dataset \
  --training_config conf/training.yaml

python run.py --stages collect_stats train infer measure \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml

python run.py --stages pack_model upload_model \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml \
  --publication_config conf/publication.yaml
```

Or run everything in one go:

```bash
python run.py \
  --stages all \
  --training_config conf/training.yaml \
  --inference_config conf/inference.yaml \
  --metrics_config conf/metrics.yaml \
  --publication_config conf/publication.yaml
```

All stage settings live in `conf/*.yaml` — there are no CLI key-value overrides.
Outputs go to `exp/<exp_tag>/` (checkpoints, logs) and `exp/<exp_tag>/inference/` (hypotheses, `metrics.json`).

::: warning
Unlike tools such as Hydra, ESPnet3 does not support overriding config values
from the CLI.
This is intentional: ESPnet3 enforces a strict **one config per experiment**
policy so that every run is fully reproducible from its YAML files alone.
If you need a different setting, edit or copy the config file — do not pass
ad-hoc flags on the command line.
:::

<details>
<summary>Advanced: import-based execution</summary>

ESPnet3 recipes are fully importable, which is useful for programmatic pipelines or MLOps workflows where you want to drive stage execution from Python rather than the CLI.

```python
from argparse import Namespace
from pathlib import Path

from egs3.TEMPLATE.asr.run import main
from espnet3.systems.asr.system import ASRSystem

args = Namespace(
    stages=["create_dataset", "collect_stats", "train", "infer", "measure"],
    training_config=Path("/path/to/training.yaml"),
    inference_config=Path("/path/to/inference.yaml"),
    metrics_config=Path("/path/to/metrics.yaml"),
    publication_config=None,
    demo_config=None,
    dry_run=False,
    write_requirements=False,
)

main(args=args, system_cls=ASRSystem)
```

</details>

## Next steps

<DocCards :cols="3">
  <DocCard
    title="What a Recipe Is"
    desc="Build a minimal recipe from scratch with your own data"
    icon="tabler:file-code"
    href="./what-is-a-recipe.html"
  />
  <DocCard
    title="Installation Troubleshooting"
    desc="CUDA mismatch, missing packages, and common errors"
    icon="tabler:tool"
    href="./installation-troubleshooting.html"
  />
  <DocCard
    title="System and Stages"
    desc="Learn how `run.py` and named stages fit together in ESPnet3."
    icon="tabler:puzzle"
    href="../stages/index.html"
  />
  <DocCard
    title="Config Overview"
    desc="See how training, inference, metrics, and publication configs connect."
    icon="tabler:settings-code"
    href="../config/index.html"
  />
</DocCards>
