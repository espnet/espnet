## ESPnet3 Configuration Guide

This document explains how to work with configuration files in ESPnet3, particularly focusing on modern usage patterns powered by OmegaConf and Hydra. The content is divided into two main parts:

1. Adoption of OmegaConf
2. Structure of the Default Recipe Configs

### 1. Adoption of OmegaConf
ESPnet3 supports OmegaConf and Hydra, enabling dynamic class instantiation, use of environment variables, and injection of runtime metadata (e.g., execution date).

### 2. Structure of the Default Recipe

The default configuration files are organized into modular sections, each responsible for a specific area of the experiment setup.

E.g., configurations that you may follow:
```yaml
# Parameters used over the experiments
expdir=
seed=
...

# Config for each elements
parallel:
   # parallel config

model:
   # model definition

trainer:
   # pytorch-lightning trainer arguments.
```

#### General Experiment Settings
Defines common items such as:
- Experiment directory
- Random seed
These are shared across multiple sub-configs to avoid redundancy.

#### Parallel Execution ("parallel")
These configs define settings for distributed execution, whether on local machines or clusters.
ESPnet3 supports both CPU and GPU parallelization via Dask.
So please check the Dask project if you want to know more on what config you can set.
- on local PC: [LocalCluster](https://docs.dask.org/en/latest/deploying-python.html?highlight=localcluster#reference)
- on Slurm cluster: [SLURMCluster](https://jobqueue.dask.org/en/latest/generated/dask_jobqueue.SLURMCluster.html)

You can define multiple parallel configs (e.g., `parallel_cpu`, `parallel_gpu`) and select one when launching your experiment.

```yaml
parallel_cpu:
  env: slurm
  n_workers: 64
  options:
    queue: cpu
    cores: 16
    processes: 4
    memory: 16GB
    walltime: 30:00

parallel_gpu:
  env: slurm
  n_workers: 3
  options:
    queue: gpu
    cores: 8
    processes: 1
    memory: 16GB
    walltime: 30:00
    job_extra_directives:
      - "--gres=gpu:1"
```

```python
from espnet3.parallel.parallel import get_client

# Submit 4 jobs, each has 16 cpus and 4 parallel jobs.
# i.e. each worker will use 4 cpus.
with get_client(config.parallel_cpu) as cpu_client:
    processed_data = parallel_map(
        <some function>,
        <data>,
        cpu_client, # you can specify client
    )

# Submit 3 jobs, each worker has 8 cpus and 1 gpu.
with get_client(config.parallel_gpu) as gpu_client:
    result = parallel_map(
        <some function>,
        <data>,
    ) # gpu_client will be used

```


#### Model Definition
ESPnet3 supports two main options for model definition:

1. **Using ESPnet-provided models**
   - You can reuse existing ESPnet2 model configs
   - Simply copy the model config and specify the ESPnet3 task to retain compatibility

   ```python
   from espnet3.trainer import LitESPnetModel
   from espnet3.task import get_espnet_model

   model = get_espnet_model(task="asr", config=config.model)
   model = LitESPnetModel(model)
   trainer = Trainer(model, ...)
   ```

2. **Using custom models defined by researchers**
   - Specify the path to the model class and its arguments in the config
   - Hydra will instantiate it at runtime

   ```python
   from espnet3.trainer import LitESPnetModel

   model = hydra.utils.instantiate(config.model)
   model = LitESPnetModel(model)
   trainer = Trainer(model, ...)
   ```

In both cases, the model is passed into the trainer in the same way.


#### Optimizer and Scheduler
ESPnet3 supports multiple optimizers and schedulers. This is useful, for example, when training GAN-based model or other complex models.
For detailed documentation please check [Multiple Optimizers and Schedulers](./optimizer_configuration.md)

#### Dataloader Configuration
Covers settings such as:
- Dataset paths and classes
- Collate functions
- Samplers

If you want to configure dataloader, please create the following section into your configuration.

E.g., simple configuration
```yaml
dataloader:
  collate_fn:
    _target_: espnet2.train.collate_fn.CommonCollateFn
    int_pad_value: -1

  train:
    shuffle: true
    batch_size: 4
    num_workers: 4

  valid:
    shuffle: false
    batch_size: 4
    num_workers: 4
```

E.g., using lhotse as dataset
```yaml
dataloader:
  collate_fn:
    _target_: espnet2.train.collate_fn.CommonCollateFn
    int_pad_value: -1

  train:
    dataset:
      _target_: lhotse.dataset.speech_recognition.K2SpeechRecognitionDataset
      input_strategy:
        _target_: lhotse.dataset.OnTheFlyFeatures
        extractor:
          _target_: lhotse.Fbank
    sampler:
      _target_: lhotse.dataset.sampling.SimpleCutSampler
      max_cuts: 20
      shuffle: true
  valid:
    dataset:
      _target_: lhotse.dataset.speech_recognition.K2SpeechRecognitionDataset
      input_strategy:
        _target_: lhotse.dataset.OnTheFlyFeatures
        extractor:
          _target_: lhotse.Fbank
    sampler:
      _target_: lhotse.dataset.sampling.SimpleCutSampler
      max_cuts: 20
      shuffle: false
```

#### Trainer Parameters
The default trainer is based on PyTorch Lightning. Most config values are passed directly into the Lightning trainer.

Some parameters require pre-instantiation (e.g., loggers and callbacks). These are instantiated during initialization and passed in properly.

The following configs will be passed to Lightning Trainer after instantiation:
- accelerator
- strategy
- logger
- profiler
- plugins
- callbacks

E.g.,
```yaml
trainer:
  # Configs directly passed to lightning trainer
  accelerator: gpu
  devices: 4
  num_nodes: 1
  accumulate_grad_batches: 1
  gradient_clip_val: 1.0
  log_every_n_steps: 500
  max_epochs: 70

  # Configs passed to lightning trainer after instantiation
  logger:
    - _target_: lightning.pytorch.loggers.TensorBoardLogger
      save_dir: ${expdir}/tensorboard
      name: tb_logger

  strategy: ddp
```

---

ESPnet3 aims to provide a flexible and modular configuration system, enabling researchers to scale from quick experiments to large-scale distributed training with minimal config changes.
