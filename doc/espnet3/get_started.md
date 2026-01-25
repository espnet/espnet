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

ESPnet3 is distributed under the **same pip package name: `espnet`**.
For more installation options (uv, pixi, source), see [ESPnet3 Installation](./install.md).

```bash
pip install espnet
```

### Install from source (recommended for development)

```bash
git clone https://github.com/espnet/espnet.git
cd espnet/tools

# Recommended: setup_uv.sh  
# Installs pixi + uv and sets up all dependencies much faster than conda.
. setup_uv.sh
```

## 📦 2. Install system-specific dependencies

ESPnet3 introduces the concept of a **system** (ASR, TTS, ST, ENH, etc.).
Each system may require additional packages not used by others.

Install system extras using:

```bash
pip install "espnet[asr]"
```

Other examples:

```bash
pip install "espnet[tts]"
pip install "espnet[st]"
pip install "espnet[enh]"
```

If installed from a cloned repository:

```bash
pip install -e ".[asr]"
# or using uv:
uv pip install -e ".[asr]"
```

## 🧪 3. Run a recipe **without cloning the repository**

(import-based execution)

ESPnet3 recipes are fully importable.
Create config files locally and run:

```python
from argparse import Namespace
from pathlib import Path

from egs3.TEMPLATE.asr.run import main
from espnet3.systems.asr.system import ASRSystem

stages = ["create_dataset", "collect_stats", "train", "infer", "measure"]
args = Namespace(
    stages=stages,
    train_config=Path("/path/to/train_config.yaml"),
    infer_config=Path("/path/to/infer_config.yaml"),
    measure_config=Path("/path/to/measure_config.yaml"),
    publish_config=None,
    demo_config=None,
    dry_run=False,
    write_requirements=False,
)

main(args=args, system_cls=ASRSystem, stages=stages)
```

This is useful for programmatic pipelines or MLOps workflows.

## 🖥 4. Run a recipe **with a cloned repository**

All configs and scripts live inside `egs3/`.

Example: LibriSpeech 100h ASR

```bash
cd egs3/librispeech_100/asr
python run.py \
    --stages all \
    --train_config conf/train.yaml \
    --infer_config conf/infer.yaml \
    --measure_config conf/measure.yaml
```

# 🧠 Understanding Stages

The default stage order is defined in:

```
egs3/TEMPLATE/<system>/run.py
```

Typical ASR pipeline:

1. **create_dataset** (download/prepare raw data)
2. **collect_stats** (compute CMVN/statistics)
3. **train** (fit the model)
4. **infer** (generate hypotheses)
5. **measure** (compute metrics)
6. **pack_model / upload_model** (package + upload artifacts)

You can run selected stages:

```bash
python run.py \
    --stages train infer measure \
    --train_config conf/train.yaml \
    --infer_config conf/infer.yaml \
    --measure_config conf/measure.yaml
```

# 🧵 Stage-specific arguments

Stages do not accept arbitrary CLI arguments. Keep all stage settings in the
YAML configs and pass the configs via `--train_config`, `--infer_config`, and
`--measure_config`.

No code changes inside the system class are needed.

# 🧩 Implementing `src/` for your recipe

Each recipe may define custom logic inside:

```
egs3/<recipe>/<task>/src/
```

Typical files:

* **create_dataset.py** - dataset preparation functions
* **dataset.py** - dataset builder or transform classes
* **custom_model.py** - user-defined modules referenced by Hydra configs

`run.py` automatically adds this directory to `PYTHONPATH`.

# ⚙️ Configurations (Hydra)

All hyperparameters live in `conf/*.yaml`.

**Important: ESPnet3 disables CLI overrides (`--key=value`).**
This is because ESPnet3 relies on hierarchical config merging that conflicts
with Hydra's runtime override semantics.
All overrides must be written inside YAML files.

# ✅ Putting Everything Together (cloned repository workflow)

Start from:

```
egs3/TEMPLATE/asr/run.py
```

Replace `ASRSystem` if you define a custom system. Then:

```bash
cd egs3/<your_recipe>/<task>

# Dataset preparation
python run.py --stages create_dataset --train_config conf/train.yaml

# (Optional) collect_stats + training
python run.py --stages collect_stats train --train_config conf/train.yaml

# Evaluation
python run.py --stages infer measure --infer_config conf/infer.yaml --measure_config conf/measure.yaml
```

Outputs go to:

* `exp/` – training logs + checkpoints
* `infer_dir/` – inference outputs + measures.json

## 📚 Additional ESPnet3 Documentation

### ✅ Cheat sheet: what you touch vs. what's provided

| Goal                        | You mainly edit / run                        | Read next                          |
| --------------------------- | -------------------------------------------- | ---------------------------------- |
| Define datasets and loaders | `conf/dataset*.yaml`, `DataOrganizer` config | [DataOrganizer and dataset pipeline](./core/components/data-organizer.md) |
| Configure training          | `conf/train.yaml` (model, trainer, optim)    | [Optimizer configuration](./core/components/optimizer_configuration.md), [Callbacks](./core/components/callbacks.md) |
| Run multi-GPU / cluster     | `conf/train.yaml` + `parallel` blocks        | [Multi-GPU / multi-node](./core/parallel/multiple_gpu.md), [Train config](./config/train_config.md) |
| Set up evaluation           | `conf/infer.yaml` + `conf/measure.yaml`      | [Inference](./stages/inference.md), [Measure](./stages/measure.md), [Provider / Runner](./core/parallel/provider_runner.md) |

### Execution Framework

* **Provider / Runner:** [Provider / Runner](./core/parallel/provider_runner.md)
* **Parallel configuration:** [Parallel](./core/parallel.md)

### Data & Datasets

* **Data preparation examples:** [Data preparation](./core/parallel/data_preparation.md)
* **Dataset classes & sharding:** [DataOrganizer and dataset pipeline](./core/components/data-organizer.md)

### Training

* **Callbacks:** [Callbacks](./core/components/callbacks.md)
* **Optimizers:** [Optimizer configuration](./core/components/optimizer_configuration.md)
* **Multiple optimizers/schedulers:** [Multiple optimizers & schedulers](./core/components/multiple_optimizers_schedulers.md)
* **Multi-GPU & multi-node:** [Multi-GPU / multi-node](./core/parallel/multiple_gpu.md)

### Inference & Evaluation

* **Runner-based decoding:** [Provider / Runner](./core/parallel/provider_runner.md)
* **Inference pipeline:** [Inference](./stages/inference.md)
* **Measurement pipeline:** [Measure](./stages/measure.md)

### Recipe Structure

* **Recipe directory layout:** [Recipe directory layout](./recipe_directory.md)
* **Systems:** [Systems](./core/systems.md)

## 💡 Tips for Working With Recipes

* Keep configs modular: dataset / model / trainer / parallel blocks.
* Decide early whether execution is **local** or **SLURM/cluster**.
* Use import-based execution for MLOps pipelines.
* Reuse ESPnet2 model configs where possible.
