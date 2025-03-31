## ESPnetEZ: Callback Mechanisms in Training

This document provides a quick overview of what is possible during training in ESPnetEZ.
Since ESPnetEZ uses PyTorch Lightning under the hood, nearly all Lightning features are available. However, this guide focuses on the unique **callbacks** that ESPnetEZ wraps and extends.

---

### 1. Checkpoint Averaging

ESPnetEZ's `trainer.py` wraps PyTorch Lightning's `Trainer` and injects callbacks during initialization. One of the key default callbacks is checkpoint averaging, which is registered automatically unless explicitly disabled.

ESPnetEZ introduces a custom callback for **checkpoint averaging** as part of the default training routine.

This callback averages the weights of multiple saved checkpoints based on a validation criterion (e.g., lowest validation loss) and saves the resulting model at the end of training.

For example, if the top 3 checkpoints (ranked by lowest validation loss) are:

- `ckpt-epoch=20.ckpt`
- `ckpt-epoch=24.ckpt`
- `ckpt-epoch=27.ckpt`

Then this callback will load these checkpoints, compute the average of their weights, and save a final model.

This process helps produce a more stable and generalizable model for inference.

---

### ✅ Configuration Example

The checkpoint averaging behavior is governed by the `best_model_criterion` setting in `config.yaml`. This specifies how checkpoints are ranked before averaging.

For example:

```yaml
best_model_criterion:
  - - valid/loss
    - 3
    - min
```

This setting is used inside `trainer.py` to determine what metric the `AverageCheckpointCallback` should monitor. The value is passed into the callback during training initialization.&#x20;

---

For more customization and details, see the implementation in `espnetez/trainer/callbacks.py`. ESPnetEZ provides a clean wrapper around PyTorch Lightning’s callback interface, making it easier to extend or adjust training behavior with minimal changes.

