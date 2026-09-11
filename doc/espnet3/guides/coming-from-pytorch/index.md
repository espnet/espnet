# Coming from PyTorch

If you already know PyTorch, most ESPnet3 pieces are still familiar.
The main shift is that ESPnet3 organizes them as recipes, Systems, and
stage-specific configs.

Use these pages to map the usual PyTorch questions:

<DocCards :cols="2">
  <DocCard
    title="Data and dataloader"
    desc="Use normal PyTorch datasets, then wire them through `dataset:` and `dataloader:`."
    icon="tabler:database"
    href="./data-and-dataloader.html"
  />
  <DocCard
    title="Model and system"
    desc="Split model code and workflow code across `src/model.py` and `System` methods."
    icon="tabler:hierarchy-2"
    href="./model-and-system.html"
  />
  <DocCard
    title="Multi-GPU"
    desc="See how Lightning training parallelism and provider-runner parallelism differ."
    icon="tabler:gpu-card"
    href="./multi-gpu.html"
  />
  <DocCard
    title="Logging and debug"
    desc="Use built-in logging, logger config, profiler config, and fast debug knobs."
    icon="tabler:bug"
    href="./logging-and-debug.html"
  />
  <DocCard
    title="Customize the training loop"
    desc="Move to recipe-local LightningModule, trainer, or System code when the default loop is not enough."
    icon="tabler:settings-cog"
    href="../finetuning/training-loop.html"
  />
</DocCards>
