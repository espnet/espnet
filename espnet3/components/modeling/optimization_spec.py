"""Structured specs for optimization configuration in ESPnet3."""

from dataclasses import dataclass
from typing import Any, Optional

import torch


@dataclass
class OptimizationStep:
    """Describe one optimizer update emitted by the model.

    ESPnet3 keeps the model return contract as `(loss, stats, weight)`. In the
    single-optimizer path, `loss` is a plain tensor. In the multiple-optimizer
    path, `loss` becomes either one `OptimizationStep` or a list of them so the
    training loop knows which named optimizer should consume each loss.

    Example:
        `OptimizationStep(loss=g_loss, name="generator")`

    means "apply `g_loss` to the optimizer and scheduler pair named
    `generator`." The `name` must match the keys configured under
    `optimizers.<name>` and `schedulers.<name>`.
    """

    loss: torch.Tensor
    name: str


@dataclass
class OptimizerSpec:
    """Describe one named optimizer block from `config.optimizers`.

    This dataclass is the normalized, validated form of one user-facing config
    entry such as:

    ```yaml
    optimizers:
      generator:
        optimizer:
          _target_: torch.optim.Adam
          lr: 0.0002
        params: generator
        accum_grad_steps: 2
        step_every_n_iters: 1
        gradient_clip_val: 1.0
        gradient_clip_algorithm: norm
    ```

    It does not store the instantiated optimizer object itself. Instead, it
    records the policy needed by the Lightning module to:
    - select which parameters belong to this optimizer,
    - decide when accumulated gradients are large enough to step,
    - decide how often this optimizer should update,
    - and apply per-optimizer gradient clipping.
    """

    name: str
    optimizer: Any
    params: str
    accum_grad_steps: int = 1
    step_every_n_iters: int = 1
    gradient_clip_val: Optional[float] = None
    gradient_clip_algorithm: str = "norm"

    @classmethod
    def from_config(cls, name: str, cfg) -> "OptimizerSpec":
        """Build and validate an optimizer spec from one named config block.

        This converts raw Hydra/OmegaConf config into a typed `OptimizerSpec`
        instance, applies dataclass defaults for omitted optional fields, and
        runs basic validation before the Lightning module instantiates the real
        optimizer objects.
        """
        if "optimizer" not in cfg:
            raise AssertionError(
                f"Optimizer '{name}' is missing nested 'optimizer' config."
            )
        if "params" not in cfg:
            raise AssertionError(f"Optimizer '{name}' is missing 'params'.")

        spec_kwargs = {
            "name": name,
            "optimizer": cfg.optimizer,
            "params": cfg.params,
        }
        if hasattr(cfg, "accum_grad_steps"):
            spec_kwargs["accum_grad_steps"] = int(cfg.accum_grad_steps)
        if hasattr(cfg, "step_every_n_iters"):
            spec_kwargs["step_every_n_iters"] = int(cfg.step_every_n_iters)
        if hasattr(cfg, "gradient_clip_val"):
            spec_kwargs["gradient_clip_val"] = cfg.gradient_clip_val
        if hasattr(cfg, "gradient_clip_algorithm"):
            spec_kwargs["gradient_clip_algorithm"] = str(cfg.gradient_clip_algorithm)

        spec = cls(**spec_kwargs)
        spec.validate()
        return spec

    def validate(self) -> None:
        """Validate update-policy settings after normalization.

        This checks only spec-local rules such as positive accumulation / step
        intervals and valid clipping algorithm names. Parameter coverage and
        overlap across different optimizers are validated later by the Lightning
        module because those checks require seeing all optimizer specs together.
        """
        if self.accum_grad_steps < 1:
            raise AssertionError(
                f"Optimizer '{self.name}' must use accum_grad_steps >= 1."
            )
        if self.step_every_n_iters < 1:
            raise AssertionError(
                f"Optimizer '{self.name}' must use step_every_n_iters >= 1."
            )
        if self.gradient_clip_algorithm not in {"norm", "value"}:
            raise AssertionError(
                f"Optimizer '{self.name}' must use gradient_clip_algorithm "
                "'norm' or 'value'."
            )


@dataclass
class SchedulerSpec:
    """Describe one named scheduler block from `config.schedulers`.

    This is the normalized, validated form of user-facing scheduler metadata
    such as:

    ```yaml
    schedulers:
      generator:
        scheduler:
          _target_: torch.optim.lr_scheduler.LinearLR
          total_iters: 1000
        interval: step
    ```

    or:

    ```yaml
    schedulers:
      discriminator:
        scheduler:
          _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
          patience: 2
        interval: epoch
        monitor: valid/discriminator/loss
    ```

    Like `OptimizerSpec`, this stores configuration, not the
    instantiated scheduler object. The Lightning module uses it to decide
    whether a scheduler should step on each optimizer update or at epoch end,
    and which logged metric to pass when a monitored epoch-based scheduler is
    used.
    """

    name: str
    scheduler: Any
    interval: str = "epoch"
    monitor: Optional[str] = None

    @classmethod
    def from_config(cls, name: str, cfg) -> "SchedulerSpec":
        """Build and validate a scheduler spec from one named config block.

        This converts raw Hydra/OmegaConf config into a typed `SchedulerSpec`
        instance, applies dataclass defaults for omitted optional fields, and
        validates interval-level rules before the Lightning module instantiates
        the real scheduler objects.
        """
        if "scheduler" not in cfg:
            raise AssertionError(
                f"Scheduler '{name}' is missing nested 'scheduler' config."
            )

        spec_kwargs = {
            "name": name,
            "scheduler": cfg.scheduler,
        }
        if hasattr(cfg, "interval"):
            spec_kwargs["interval"] = str(cfg.interval)
        if hasattr(cfg, "monitor"):
            spec_kwargs["monitor"] = cfg.monitor

        spec = cls(**spec_kwargs)
        spec.validate()
        return spec

    def validate(self) -> None:
        """Validate scheduler stepping policy after normalization.

        Only spec-local checks are performed here. Cross-object rules, such as
        matching optimizer and scheduler names exactly, are validated later by
        the Lightning module once all specs have been collected.
        """
        if self.interval not in {"step", "epoch"}:
            raise AssertionError(
                f"Scheduler '{self.name}' must use interval 'step' or 'epoch'."
            )


@dataclass
class OptimizerRuntimeState:
    """Track custom runtime counters for one named optimizer.

    Lightning already checkpoints optimizer and scheduler `state_dict()` values.
    This dataclass exists only for extra ESPnet3 runtime state that Lightning
    does not manage for named multi-optimizer training:
    - `accum_counter`: how many backward passes have been accumulated since the
      last optimizer step,
    - `update_step`: how many times this optimizer has actually been updated.

    These counters are saved and restored through `on_save_checkpoint()` and
    `on_load_checkpoint()` in the Lightning module.
    """

    accum_counter: int = 0
    update_step: int = 0
