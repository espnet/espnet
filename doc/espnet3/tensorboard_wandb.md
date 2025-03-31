## ESPnetEZ: Logging with TensorBoard and Wandb

In ESPnet2, TensorBoard and Wandb were integrated manually for experiment logging. However, ESPnetEZ relies on **PyTorch Lightning's built-in logging support**, allowing users to configure loggers directly in the YAML config files.

As a result, ESPnetEZ does **not carry over the legacy implementation** from ESPnet2. Instead, users are encouraged to take full advantage of the logging systems already supported by PyTorch Lightning.

We **recommend using either [TensorBoard](https://www.tensorflow.org/tensorboard) or [Wandb](https://wandb.ai/site/)** for monitoring your experiments. These tools provide dashboards for tracking:
- Training and validation loss
- Accuracy and other metrics
- Learning rate schedules
- And more

---

### ✅ Example: Logger Configuration in `config.yaml`

```yaml
trainer:
  logger:
    - _target_: lightning.pytorch.loggers.TensorBoardLogger
      save_dir: ${expdir}/tensorboard
      name: tb_logger

    - _target_: lightning.pytorch.loggers.WandbLogger
      name: espnetez-experiment
      save_dir: ${expdir}/wandb
      project: espnetez-project
```

This will simultaneously log to both TensorBoard and Wandb.

For more details, refer to the official PyTorch Lightning documentation:
- [TensorBoardLogger](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.TensorBoardLogger.html)
- [WandbLogger](https://lightning.ai/docs/pytorch/stable/extensions/generated/lightning.pytorch.loggers.WandbLogger.html)

---

By configuring loggers via YAML and using Lightning’s built-in support, ESPnetEZ simplifies monitoring and improves compatibility with popular experiment tracking platforms.

