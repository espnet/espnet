## ESPnet3 Design Overview

This document outlines the design principles and architecture of espnet3. Compared to ESPnet2, ESPnet3 emphasizes reproducibility, flexibility, and maintainability, aiming to streamline and accelerate research in speech processing.

### 1. Dataset Preparation and Sharing

#### Background
Reproducibility becomes difficult when preprocessed datasets are not shared. In cluster environments, having every user reprocess the same data is inefficient.

#### Policy
- Promote use of Hugging Face Datasets for standardized formats and sharing
- Support custom formats on clusters (e.g., JSON, Kaldi scp files)
- `ESPnetDataset` is a wrapper class designed to match the output format expected by ESPnet models. For example, it is compatible with `CommonCollateFn`. However, its usage is not mandatory—it's simply a recommended option.

#### Code Example (from ESPnet3/data/dataset.py)
```python
from datasets import load_dataset
from espnet3.data.dataset import ESPnet3Dataset

# Load HuggingFace dataset
hf_dataset = load_dataset("some_dataset", split="train")

# Wrap for ESPnet3
espnet_dataset = ESPnet3Dataset(hf_dataset)
```

### 2. Dataloader and Batching Strategy

#### Challenge
Speech data varies significantly in sequence length. Standard DataLoaders often introduce large padding, leading to inefficient training.

#### Solution
- Use Lhotse for on-the-fly feature extraction and length-based sorting
- Support dynamic batch sizing
- HuggingFace Datasets are supported via custom integration

#### Code Example (from egs3/librispeech/asr1/data.py)
```python
from lhotse.dataset import DynamicBucketingSampler
from espnet3.data.loader import make_dataloader

dataloader = make_dataloader(dataset, sampler=DynamicBucketingSampler(...))
```

### 3. Model and Trainer Definition

#### Improvements
- ESPnet2 required full parameter specification in config files, making maintenance difficult
- ESPnet3 adopts OmegaConf for modern, flexible configuration with partial overrides

#### Trainer
- Based on PyTorch Lightning: `ESPnet3LightningTrainer`
- Training can be launched simply by passing a model and config

#### Code Example (from ESPnet3/trainer/trainer.py)
```python
from espnet3.trainer.trainer import ESPnet3LightningTrainer
from espnet3.trainer.model import LitESPnetModel

model = LitESPnetModel(task)
trainer = ESPnet3LightningTrainer(model=model, config=config, expdir="exp")
trainer.fit()
```

### 4. Customization and Extensibility

#### Background
ESPnet2 had tightly coupled models and trainers, making customization difficult.

#### Solution
- Clear separation between the model (`LitESPnetModel`) and the trainer (`ESPnet3LightningTrainer`)
- Callbacks, loggers, and schedulers are fully configurable through YAML
- Leverages PyTorch Lightning's extensibility and community ecosystem

#### Code Example (from ESPnet3/trainer/callbacks.py)
```python
from espnet3.trainer.callbacks import get_default_callbacks

callbacks = get_default_callbacks(config)
```

---

In summary, ESPnet3 is designed for reproducible, efficient, and flexible research workflows. It remains compatible with ESPnet2 recipes and serves as a robust foundation for future developments in speech processing.

