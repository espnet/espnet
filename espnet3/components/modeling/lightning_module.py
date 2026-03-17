"""ESPnet3 PyTorch LightningModule for training and data integration."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import lightning
import torch
from humanfriendly import format_number, format_size
from hydra.utils import instantiate
from omegaconf import OmegaConf

from espnet2.train.collate_fn import CommonCollateFn
from espnet3.components.data.collect_stats import collect_stats
from espnet3.components.data.dataloader import DataLoaderBuilder
from espnet3.components.modeling.optimization_spec import (
    OptimizationStep,
    OptimizerRuntimeState,
    OptimizerSpec,
    SchedulerSpec,
)
from espnet3.utils.logging_utils import log_component, log_stage

logger = logging.getLogger("lightning")


class ESPnetLightningModule(lightning.LightningModule):
    """ESPnet3 LightningModule wrapper for model training and data integration.

    This wrapper keeps the common ESPnet3 model contract unchanged:

    ```python
    loss, stats, weight = model(**batch)
    ```

    Most models should continue to return a single scalar loss tensor. The training
    loop then behaves exactly like conventional Lightning single-optimizer training.

    When multiple optimizers are configured, the same return value is expected,
    but the `loss` field must carry optimizer routing information through
    `OptimizationStep`.
    The model still returns one `stats` dict and one optional `weight` value; only
    the type of `loss` changes.

    Example:
        Single optimizer path:
        ```python
        def forward(self, **batch):
            loss = ...
            stats = {"loss": loss.detach(), "acc": acc.detach()}
            weight = torch.tensor(batch_size, device=loss.device)
            return loss, stats, weight
        ```

        GAN-style path updating both optimizers in a single batch:
        ```python
        def forward(self, **batch):
            g_loss = ...
            d_loss = ...
            stats = {
                "generator_loss": g_loss.detach(),
                "discriminator_loss": d_loss.detach(),
            }
            return [
                OptimizationStep(loss=g_loss, name="generator"),
                OptimizationStep(loss=d_loss, name="discriminator"),
            ], stats, None
        ```

        GAN-style path updating only the generator for one batch:
        ```python
        def forward(self, **batch):
            g_loss = ...
            stats = {"generator_loss": g_loss.detach()}
            return OptimizationStep(loss=g_loss, name="generator"), stats, None
        ```

    Notes:
        - Returning `OptimizationStep` with the single optimizer is forbidden.
        - In the multi-optimizer path, only optimizers named by returned
          `OptimizationStep` objects are touched for that batch. Optimizers omitted
          from the list are left untouched entirely.
        - The order of `OptimizationStep` entries is the exact backward/step order.
        - NaN or Inf in any returned loss causes the whole batch to be skipped on
          all workers.
    """

    def __init__(self, model, config):
        """Initialize the ESPnet LightningModule wrapper."""
        super().__init__()
        self.config = config
        self.model = model
        data_organizer = instantiate(config.dataset)

        data_organizer.log_summary(logger)
        self.train_dataset = data_organizer.train
        self.valid_dataset = data_organizer.valid
        self.nan_countdown = 0

        self._optimizer_specs: List[OptimizerSpec] = []
        self._scheduler_specs: List[SchedulerSpec] = []
        self._optimizer_states: Dict[str, OptimizerRuntimeState] = {}
        self._multi_optimizer_names: List[str] = []
        self._named_optimizers_cache: Optional[Dict[str, torch.optim.Optimizer]] = None
        self._named_schedulers_cache: Optional[Dict[str, object]] = None

        # If user is trying to use both Pytorch dataloader and ESPnet's dataloader
        # then raise an error here.
        is_train_espnet = False
        if (
            hasattr(self.config.dataloader.train, "iter_factory")
            and self.config.dataloader.train.iter_factory is not None
        ):
            is_train_espnet = True

        is_valid_espnet = False
        if (
            hasattr(self.config.dataloader.valid, "iter_factory")
            and self.config.dataloader.valid.iter_factory is not None
        ):
            is_valid_espnet = True

        assert (
            is_train_espnet == is_valid_espnet
        ), "Train and valid should have the same type of dataloader."

        self.is_espnet_sampler = is_train_espnet

        self.collate_fn = CommonCollateFn(int_pad_value=-1)
        if hasattr(self.config.dataloader, "collate_fn"):
            self.collate_fn = instantiate(self.config.dataloader.collate_fn)

        # Named `optimizers` switches the module to the manual multi-optimizer path.
        self.automatic_optimization = getattr(self.config, "optimizers", None) is None

    def _sync2skip(self, flag_skip):
        """Synchronize a skip flag across all DDP workers.

        Args:
            flag_skip (torch.Tensor): Boolean scalar indicating whether this worker
                should skip.

        Returns:
            bool: True if any worker flags a skip.
        """
        # see below:
        # https://github.com/Lightning-AI/lightning/issues/5243#issuecomment-1552650013
        # gathering a tensor across all workers and then reduce it using or
        if self.config.num_device == 1:
            any_invalid = flag_skip
        else:
            world_size = torch.distributed.get_world_size()
            torch.distributed.barrier()

            result = [torch.zeros_like(flag_skip) for _ in range(world_size)]
            torch.distributed.all_gather(result, flag_skip)

            any_invalid = torch.sum(torch.stack(result)).bool().item()

        return any_invalid

    def _check_nan_inf_loss(
        self, losses: Sequence[torch.Tensor], batch_id: int
    ) -> bool:
        """Check one or more losses for NaN/Inf and synchronize a full batch skip.

        This method is shared by both optimization modes:

        - Single-optimizer-path mode, where the model returns one scalar loss tensor.
        - Multiple-path mode, where the model returns one or more
          `OptimizationStep` objects and therefore several loss tensors can be
          produced in the same batch.

        If every loss is finite, the batch proceeds normally. If any one loss is
        NaN or Inf, the entire batch is skipped across all workers. The multi-loss
        path intentionally does not allow partial updates because generator /
        discriminator style training must keep all workers synchronized and avoid
        updating only a subset of returned losses.

        Args:
            losses: Sequence of loss tensors produced for the current batch.
                This is a one-element sequence in the single-optimizer-path
                case and one tensor per returned optimization step in the
                multiple-path case.
            batch_id: Batch index used in warning messages.

        Returns:
            bool: True if any worker observed an invalid loss and the batch should
            be skipped globally.
        """
        invalid = False
        for loss in losses:
            mask_nan_inf = torch.logical_or(torch.isnan(loss), ~torch.isfinite(loss))
            if torch.any(mask_nan_inf):
                invalid = True
                break

        flag_skip = torch.tensor(
            invalid,
            device=losses[0].device if losses else self.device,
            dtype=torch.bool,
        )

        any_invalid = self._sync2skip(flag_skip)
        if any_invalid:
            if self.nan_countdown >= 100:
                raise RuntimeError(
                    "Too many NaNs loss iterations encountered, stopping!"
                )
            logger.warning(
                f"NaN loss in batch {batch_id} of epoch {self.current_epoch}, "
                f"skipping the whole batch across all workers."
            )
            self.nan_countdown += 1
        else:
            self.nan_countdown = 1

        return any_invalid

    def _validate_multi_loss_steps(self, loss) -> List[OptimizationStep]:
        """Validate and normalize multi-optimizer loss output into step objects.

        Checks:
            - `loss` is either one `OptimizationStep` or a list of them.
            - The list is not empty.
            - Every list element is an `OptimizationStep`.

        Returns:
            List[OptimizationStep]: Ordered optimization steps. A single
            `OptimizationStep` is normalized to a one-element list.
        """
        if isinstance(loss, OptimizationStep):
            steps = [loss]
        elif isinstance(loss, list):
            steps = loss
        else:
            steps = None

        if steps is None:
            raise AssertionError(
                "Multiple optimizers are configured, so `loss` must be "
                "`OptimizationStep` or `list[OptimizationStep]` to specify which "
                "optimizer each loss updates."
            )
        if len(steps) == 0:
            raise AssertionError(
                "Multiple optimizers are configured, but the model returned an empty "
                "optimization step list."
            )
        for step in steps:
            if not isinstance(step, OptimizationStep):
                raise AssertionError(
                    "Multiple optimizers are configured, so every item in `loss` "
                    "must be an `OptimizationStep`."
                )
        return steps

    def _log_stats(
        self,
        mode: str,
        stats: Dict[str, torch.Tensor],
        weight,
        extra_stats: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Log stats dict plus optional extra metrics while tolerating weight=None."""
        if not isinstance(stats, dict):
            raise AssertionError(
                "Model output `stats` must be a dict so metrics can be logged "
                "consistently."
            )

        new_stats = {}
        for k, v in stats.items():
            if v is not None:
                new_stats[f"{mode}/{k}"] = v.item() if hasattr(v, "item") else v

        if extra_stats is not None:
            for k, v in extra_stats.items():
                new_stats[f"{mode}/{k}"] = v.item() if hasattr(v, "item") else v

        if getattr(self, "_trainer", None) is None:
            return

        log_kwargs = dict(
            prog_bar=True,
            logger=True,
            sync_dist=(mode == "valid"),
        )
        if weight is not None:
            log_kwargs["batch_size"] = (
                weight.item() if hasattr(weight, "item") else weight
            )
        self.log_dict(new_stats, **log_kwargs)

    def _build_optimizer_specs(self) -> List[OptimizerSpec]:
        """Build typed specs for validating the named multi-optimizer config.

        This helper is only used when `config.optimizers` enables the multiple
        optimizer path. It converts raw user config entries into `OptimizerSpec`
        objects so that the remaining setup code can perform structured validation
        and instantiation.
        """
        return [
            OptimizerSpec.from_config(name, cfg)
            for name, cfg in self.config.optimizers.items()
        ]

    def _build_scheduler_specs(self) -> List[SchedulerSpec]:
        """Build typed specs for validating the named multi-scheduler config.

        This helper is only used when `config.schedulers` is provided for the
        multiple optimizer path. It converts raw scheduler config entries into
        `SchedulerSpec` objects so that the remaining setup code can perform
        structured validation and instantiation.
        """
        if getattr(self.config, "schedulers", None) is None:
            return []

        return [
            SchedulerSpec.from_config(name, cfg)
            for name, cfg in self.config.schedulers.items()
        ]

    def _validate_named_optimizer_configs(self, specs: List[OptimizerSpec]) -> None:
        """Validate that named optimizers cover trainable parameters exactly once.

        This check is only used in the multiple optimizer path. Its goal is to
        ensure that every trainable parameter is assigned to exactly one named
        optimizer before any optimizer objects are instantiated.

        Concretely, it checks that:
            - every optimizer spec matches at least one trainable parameter
            - no trainable parameter is matched by more than one optimizer spec
            - every trainable parameter is covered by some optimizer spec

        The matching itself is based on the configured `params` substring selector.
        """
        trainable_params = {
            name: param
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        used_param_names = set()

        for spec in specs:
            selected_names = [
                name for name in trainable_params.keys() if spec.params in name
            ]
            if len(selected_names) == 0:
                raise AssertionError(
                    f"No trainable parameters found for optimizer '{spec.name}' "
                    f"with params selector '{spec.params}'."
                )
            for name in selected_names:
                if name in used_param_names:
                    raise AssertionError(
                        f"Parameter {name} is assigned to multiple optimizers."
                    )
                used_param_names.add(name)

        uncovered = set(trainable_params.keys()) - used_param_names
        if uncovered:
            raise AssertionError(
                f"{sorted(uncovered)} are not assigned to any optimizer."
            )

    def _instantiate_named_optimizers(
        self, specs: List[OptimizerSpec]
    ) -> List[torch.optim.Optimizer]:
        """Instantiate optimizers after validating param selectors."""
        trainable_params = {
            name: param
            for name, param in self.named_parameters()
            if param.requires_grad
        }
        optimizers = []
        for spec in specs:
            selected = [
                param for name, param in trainable_params.items() if spec.params in name
            ]
            optimizers.append(
                instantiate(
                    OmegaConf.to_container(spec.optimizer, resolve=True),
                    selected,
                )
            )
        return optimizers

    def configure_optimizers(self):
        """Configure single-optimizer-path or named multi-path optimizers.

        This includes the paired scheduler configuration.

        Single-optimizer-path training keeps the traditional ESPnet contract:

        ```yaml
        optimizer:
          _target_: torch.optim.Adam
          lr: 0.001

        scheduler:
          _target_: torch.optim.lr_scheduler.StepLR
          step_size: 10
          gamma: 0.5

        scheduler_interval: step
        ```

        The model keeps returning a plain tensor loss:

        ```python
        def forward(self, **batch):
            loss = ...
            stats = {"loss": loss.detach(), "acc": acc.detach()}
            weight = torch.tensor(batch_size, device=loss.device)
            return loss, stats, weight
        ```

        Single-optimizer-path schedulers follow standard Lightning behavior.
        Example with a validation-monitored `ReduceLROnPlateau`:

        ```yaml
        optimizer:
          _target_: torch.optim.Adam
          lr: 0.001

        scheduler:
          _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
          patience: 2
          factor: 0.5

        scheduler_interval: epoch
        scheduler_monitor: valid/loss
        ```

        In this case Lightning receives:
        - `interval="epoch"`
        - `monitor="valid/loss"`

        so it steps the scheduler after validation using the logged `valid/loss`
        metric.

        Multiple-path training is enabled by configuring named `optimizers` and
        `schedulers`. The names become the routing keys used by
        `OptimizationStep(name=...)`.

        ```yaml
        optimizers:
          generator:
            optimizer:
              _target_: torch.optim.Adam
              lr: 0.0002
            params: generator
            accum_grad_steps: 1
            step_every_n_iters: 1
            gradient_clip_val: 1.0
            gradient_clip_algorithm: norm

          discriminator:
            optimizer:
              _target_: torch.optim.Adam
              lr: 0.0002
            params: discriminator
            accum_grad_steps: 1
            step_every_n_iters: 1

        schedulers:
          generator:
            scheduler:
              _target_: torch.optim.lr_scheduler.LinearLR
              start_factor: 1.0
              end_factor: 0.5
              total_iters: 1000
            interval: step

          discriminator:
            scheduler:
              _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
              patience: 2
              factor: 0.5
            interval: epoch
            monitor: valid/discriminator/loss
        ```

        The matching model return may update both branches in one batch:

        ```python
        return [
            OptimizationStep(loss=g_loss, name="generator"),
            OptimizationStep(loss=d_loss, name="discriminator"),
        ], {
            "generator_loss": g_loss.detach(),
            "discriminator_loss": d_loss.detach(),
        }, None
        ```

        Or update only one branch:

        ```python
        return OptimizationStep(loss=g_loss, name="generator"), {
            "generator_loss": g_loss.detach(),
        }, None
        ```

        Important rules:
        - Single-optimizer-path training must return a tensor loss directly.
        - Multiple-path training must return `OptimizationStep` or
          `list[OptimizationStep]` as `loss` so that ESPnet3 knows which optimizer
          should be used to update parameters.
        - Optimizer and scheduler names must match exactly.
          Valid:
          ```yaml
          optimizers: {generator: {...}, discriminator: {...}}
          schedulers: {generator: {...}, discriminator: {...}}
          ```
          Error example:
          ```yaml
          optimizers: {generator: {...}, discriminator: {...}}
          schedulers: {generator: {...}, decoder: {...}}
          ```
        - In the multiple-path configuration, gradient clipping is configured per
          optimizer via `gradient_clip_val` and `gradient_clip_algorithm`.
          Trainer-level global clipping settings must not be used.
        - `monitor` names must refer to logged metric keys of the form
          `train/<stats-key>` or `valid/<stats-key>`, including automatically
          logged multi-path losses such as `valid/generator/loss` and
          `valid/discriminator/loss`.
        - `monitor` is only used with epoch-based schedulers. Step-based schedulers
          are always called as `scheduler.step()` after a successful optimizer
          update and do not receive metric inputs.
        - DeepSpeed is rejected in the multi-path configuration because Lightning's
          DeepSpeed strategy does not support multiple optimizers/schedulers.
        """
        if getattr(self.config, "optimizer", None) and getattr(
            self.config, "scheduler", None
        ):
            assert (
                getattr(self.config, "optimizers", None) is None
            ), "Mixture of `optimizer` and `optimizers` is not allowed."
            assert (
                getattr(self.config, "schedulers", None) is None
            ), "Mixture of `scheduler` and `schedulers` is not allowed."

            params = filter(lambda p: p.requires_grad, self.parameters())
            optimizer = instantiate(
                OmegaConf.to_container(self.config.optimizer, resolve=True), params
            )
            scheduler = instantiate(
                OmegaConf.to_container(self.config.scheduler, resolve=True),
                optimizer=optimizer,
            )
            interval = str(getattr(self.config, "scheduler_interval", "step"))
            if interval not in {"step", "epoch"}:
                raise AssertionError(
                    "Single-optimizer-path `scheduler_interval` must be "
                    "'step' or 'epoch'."
                )
            monitor = getattr(self.config, "scheduler_monitor", None)
            config = {
                "scheduler": scheduler,
                "interval": interval,
            }
            if monitor is not None:
                config["monitor"] = monitor
            return {"optimizer": optimizer, "lr_scheduler": config}

        if getattr(self.config, "optimizers", None) and getattr(
            self.config, "schedulers", None
        ):
            assert (
                getattr(self.config, "optimizer", None) is None
            ), "Mixture of `optimizer` and `optimizers` is not allowed."
            assert (
                getattr(self.config, "scheduler", None) is None
            ), "Mixture of `scheduler` and `schedulers` is not allowed."
            if getattr(self.config, "scheduler_interval", None) is not None:
                raise AssertionError(
                    "Top-level `scheduler_interval` is only supported in the "
                    "single-optimizer path. Use `schedulers.<name>.interval` "
                    "for multiple optimizers."
                )
            if getattr(self.config, "scheduler_monitor", None) is not None:
                raise AssertionError(
                    "Top-level `scheduler_monitor` is only supported in the "
                    "single-optimizer path. Use `schedulers.<name>.monitor` "
                    "for multiple optimizers."
                )

            # Normalize and validate the named multi-optimizer/scheduler config first.
            self._optimizer_specs = self._build_optimizer_specs()
            self._scheduler_specs = self._build_scheduler_specs()

            optimizer_names = {spec.name for spec in self._optimizer_specs}
            scheduler_names = {spec.name for spec in self._scheduler_specs}
            if optimizer_names != scheduler_names:
                raise AssertionError(
                    "Optimizer and scheduler names must match exactly: "
                    f"optimizers={sorted(optimizer_names)}, "
                    f"schedulers={sorted(scheduler_names)}"
                )

            self._validate_named_optimizer_configs(self._optimizer_specs)
            optimizers = self._instantiate_named_optimizers(self._optimizer_specs)
            schedulers = []

            optimizer_by_name = {
                spec.name: optimizer
                for spec, optimizer in zip(self._optimizer_specs, optimizers)
            }
            for spec in self._scheduler_specs:
                scheduler = instantiate(
                    OmegaConf.to_container(spec.scheduler, resolve=True),
                    optimizer=optimizer_by_name[spec.name],
                )
                schedulers.append(scheduler)

            self._multi_optimizer_names = [spec.name for spec in self._optimizer_specs]
            self._optimizer_states = {
                spec.name: OptimizerRuntimeState() for spec in self._optimizer_specs
            }
            self._named_optimizers_cache = None
            self._named_schedulers_cache = None
            return optimizers, schedulers

        raise ValueError(
            "Must specify either `optimizer` or `optimizers` and `scheduler` or"
            "`schedulers`"
        )

    @staticmethod
    def _log_training_summary(
        logger: logging.Logger,
        model,
        optimizer=None,
        scheduler=None,
    ) -> None:
        """Log model/optimizer/scheduler details for training runs.

        Description:
            Emits a compact summary of the model, parameter counts, dtype
            composition, and optimizer/scheduler configuration. This mirrors the
            previous utility implementation but keeps logging close to where
            the training loop is configured. The method accepts either a single
            optimizer/scheduler or a list of instantiated objects from the named
            multi-optimizer path and logs them in order.

        Args:
            logger (logging.Logger): Logger used to emit messages.
            model: PyTorch model to summarize.
            optimizer: Optimizer instance or list of optimizer instances.
            scheduler: Scheduler instance or list of scheduler instances.

        Returns:
            None

        Notes:
            - Uses `format_number` and `format_size` to improve readability.
            - The summary is logged at INFO level.

        Examples:
            ```python
            self._log_training_summary(logger, self.model, optimizer, scheduler)
            ```

            Sample output:
            ```
            Model summary:
                Class Name: DummyModel
                Total Number of model parameters: 1,024
                Trainable model parameters: 1,024 (100.0%)
                Model size: 4.0 KB
                DType composition: torch.float32(100.0%)
            Optimizer[0]:
            Scheduler[0]:
            ```
            When multiple optimizers or schedulers are configured, additional
            `Optimizer[i]` and `Scheduler[i]` sections are emitted for each item in
            the instantiated list.

        Raises:
            None
        """
        logger.log(logging.INFO, "Model:\n%r", model, stacklevel=2)

        params = list(model.parameters())
        total_params = sum(p.numel() for p in params)
        trainable_params = sum(p.numel() for p in params if p.requires_grad)
        size_bytes = sum(p.numel() * p.element_size() for p in params)

        dtype_counts: dict[str, int] = {}
        for p in params:
            dtype_counts[str(p.dtype)] = dtype_counts.get(str(p.dtype), 0) + p.numel()
        dtype_items = sorted(dtype_counts.items(), key=lambda kv: kv[1], reverse=True)
        dtype_desc = ", ".join(
            f"{k}({v / total_params * 100:.1f}%)" for k, v in dtype_items
        )

        logger.log(logging.INFO, "Model summary:", stacklevel=2)
        logger.log(
            logging.INFO, "    Class Name: %s", type(model).__name__, stacklevel=2
        )
        logger.log(
            logging.INFO,
            "    Total Number of model parameters: %s",
            format_number(total_params),
            stacklevel=2,
        )
        logger.log(
            logging.INFO,
            "    Trainable model parameters: %s (%.1f%%)",
            format_number(trainable_params),
            (trainable_params / total_params * 100.0) if total_params else 0.0,
            stacklevel=2,
        )
        logger.log(
            logging.INFO,
            "    Model size: %s",
            format_size(size_bytes),
            stacklevel=2,
        )
        logger.log(logging.INFO, "    DType composition: %s", dtype_desc, stacklevel=2)

        if optimizer is None and scheduler is None:
            return

        optimizers = optimizer if isinstance(optimizer, list) else [optimizer]
        for idx, optim in enumerate(optimizers):
            logger.log(logging.INFO, "Optimizer[%d]:", idx, stacklevel=2)
            log_component(
                logger, kind="Optimizer", label=str(idx), obj=optim, max_depth=2
            )

        if scheduler is None:
            return
        schedulers = scheduler if isinstance(scheduler, list) else [scheduler]
        for idx, sch in enumerate(schedulers):
            logger.log(logging.INFO, "Scheduler[%d]:", idx, stacklevel=2)
            log_component(
                logger, kind="Scheduler", label=str(idx), obj=sch, max_depth=2
            )

    def _get_named_optimizers(self) -> Dict[str, torch.optim.Optimizer]:
        """Map configured optimizer names to instantiated optimizer objects.

        Lightning returns optimizers as a positional collection via
        `self.optimizers(use_pl_optimizer=True)`, while ESPnet3's multiple-optimizer
        path routes updates by name (for example `"generator"` or
        `"discriminator"`). This helper bridges the two representations by
        rebuilding a `name -> optimizer` mapping using the configured optimizer order.

        Example:
            ```python
            steps = [OptimizationStep(loss=g_loss, name="generator")]
            named_optimizers = self._get_named_optimizers()
            generator_optimizer = named_optimizers["generator"]
            generator_optimizer.step()
            # "discriminator" is not touched in this batch.
            ```
        """
        if self._named_optimizers_cache is not None:
            return self._named_optimizers_cache

        optimizers = self.optimizers(use_pl_optimizer=True)
        if not isinstance(optimizers, (list, tuple)):
            optimizers = [optimizers]
        self._named_optimizers_cache = dict(
            zip(self._multi_optimizer_names, optimizers)
        )
        return self._named_optimizers_cache

    def _get_named_schedulers(self) -> Dict[str, object]:
        """Map configured scheduler names to instantiated scheduler objects.

        Like optimizers, Lightning exposes schedulers positionally, while ESPnet3's
        multiple-optimizer training loop needs named access so that each optimizer
        step can trigger the scheduler with the same name. This helper rebuilds a
        `name -> scheduler` mapping from Lightning's scheduler collection.

        Example:
            ```python
            named_schedulers = self._get_named_schedulers()
            generator_scheduler = named_schedulers["generator"]
            generator_scheduler.step()
            # "discriminator" is not stepped in this batch.
            ```
        """
        if self._named_schedulers_cache is not None:
            return self._named_schedulers_cache

        schedulers = self.lr_schedulers()
        if not isinstance(schedulers, (list, tuple)):
            schedulers = [schedulers]
        self._named_schedulers_cache = dict(
            zip(self._multi_optimizer_names, schedulers)
        )
        return self._named_schedulers_cache

    def _step_named_scheduler_on_update(self, name: str) -> None:
        """Step a named scheduler on optimizer update when configured for step mode."""
        scheduler_by_name = self._get_named_schedulers()
        spec_by_name = {spec.name: spec for spec in self._scheduler_specs}
        scheduler = scheduler_by_name[name]
        spec = spec_by_name[name]
        if spec.interval != "step":
            return
        if spec.monitor is not None:
            raise AssertionError(
                f"Step-based scheduler '{name}' must not define `monitor`; "
                "metric-based schedulers are stepped at epoch end."
            )
        scheduler.step()

    def _run_multi_optimizer_updates(
        self,
        optimizer_steps: List[OptimizationStep],
        stats: Dict[str, torch.Tensor],
        weight,
        batch_idx: int,
    ) -> None:
        """Run named optimization steps with manual optimization.

        This method also updates per-optimizer runtime state.

        This path exists to support GAN and other multi-loss training without
        introducing a new model hook. The model still returns `(loss, stats, weight)`.
        Only the `loss` field changes shape:

        - single optimizer path: a tensor
        - multiple path: `OptimizationStep` or `list[OptimizationStep]`

        The `stats` dict remains a single shared logging payload. In the multi-path
        case this method additionally logs `train/<name>/loss` and
        `train/<name>/update_step` for each returned optimization step.
        """
        named_optimizers = self._get_named_optimizers()
        spec_by_name = {spec.name: spec for spec in self._optimizer_specs}

        extra_stats: Dict[str, torch.Tensor] = {}
        for step in optimizer_steps:
            if step.name not in spec_by_name:
                raise AssertionError(
                    f"Unknown optimizer '{step.name}'. Available optimizers: "
                    f"{', '.join(sorted(spec_by_name))}."
                )
            spec = spec_by_name[step.name]
            state = self._optimizer_states[step.name]
            optimizer = named_optimizers[step.name]

            if state.accum_counter == 0:
                optimizer.zero_grad()

            self.manual_backward(step.loss / spec.accum_grad_steps)
            state.accum_counter += 1
            extra_stats[f"{step.name}/loss"] = step.loss.detach()

            meets_accum = state.accum_counter >= spec.accum_grad_steps
            meets_iter = (batch_idx + 1) % spec.step_every_n_iters == 0
            if not (meets_accum and meets_iter):
                continue

            if spec.gradient_clip_val is not None:
                self.clip_gradients(
                    optimizer,
                    gradient_clip_val=spec.gradient_clip_val,
                    gradient_clip_algorithm=spec.gradient_clip_algorithm,
                )

            optimizer.step()
            optimizer.zero_grad()
            state.accum_counter = 0
            state.update_step += 1
            extra_stats[f"{step.name}/update_step"] = float(state.update_step)
            self._step_named_scheduler_on_update(step.name)

        self._log_stats("train", stats, weight, extra_stats=extra_stats)

    def _step(self, batch, batch_idx, mode):
        """Run one train/valid iteration for single or multiple optimizer modes.

        Expected model return:
        - Single-optimizer-path training or validation:
          `loss: torch.Tensor, stats: dict, weight: Optional[Tensor]`
        - Multiple-optimizer training or validation:
          `loss: OptimizationStep | list[OptimizationStep], stats: dict,
          weight: Optional[Tensor]`

        Training behavior differs between the two paths:
        - Single optimizer path keeps Lightning automatic optimization enabled.
          This method
          only prepares and returns the loss tensor, and Lightning performs the
          backward pass, optimizer step, and scheduler step after `training_step`.
        - Multiple-optimizer path uses manual optimization. This method validates
          the returned `OptimizationStep` objects and performs backward,
          optimizer stepping, clipping, and step-based scheduler updates directly.
        """
        loss, stats, weight = self.model(**batch[1])
        # Single-optimizer-path training delegates optimization to Lightning automatic
        # optimization after `training_step` returns the loss tensor. Only the
        # named multi-optimizer path performs manual optimizer handling here.
        if getattr(self.config, "optimizers", None) is not None:
            optimizer_steps = self._validate_multi_loss_steps(loss)
            any_invalid = self._check_nan_inf_loss(
                [step.loss for step in optimizer_steps], batch_idx
            )
            if any_invalid:
                return None

            if mode == "train":
                self._run_multi_optimizer_updates(
                    optimizer_steps, stats, weight, batch_idx
                )
            else:
                extra_stats = {
                    f"{step.name}/loss": step.loss.detach() for step in optimizer_steps
                }
                self._log_stats(mode, stats, weight, extra_stats=extra_stats)
            return None

        if not isinstance(loss, torch.Tensor):
            raise AssertionError(
                "Single-optimizer training expects `loss` to be a tensor. If only "
                "one loss is needed, return it directly instead of wrapping it in "
                "`OptimizationStep`."
            )

        any_invalid = self._check_nan_inf_loss([loss], batch_idx)
        if any_invalid:
            return None

        self._log_stats(mode, stats, weight)
        return loss

    def training_step(self, batch, batch_idx):
        """Training step logic."""
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        """Run the validation step logic."""
        return self._step(batch, batch_idx, mode="valid")

    def on_train_epoch_end(self) -> None:
        """Step epoch-based schedulers after metrics have been aggregated.

        This hook primarily exists for the multiple-loss / multiple-optimizer
        path, where ESPnet3 owns the optimizer and scheduler orchestration. The
        single-optimizer path keeps Lightning automatic optimization enabled, so
        Lightning handles epoch-end scheduler stepping there.
        """
        if getattr(self.config, "optimizers", None) is None:
            # Single-optimizer-path scheduler stepping is delegated to Lightning
            # automatic optimization, so this hook only handles named multi-path
            # schedulers.
            return

        scheduler_by_name = self._get_named_schedulers()
        for spec in self._scheduler_specs:
            if spec.interval != "epoch":
                continue
            scheduler = scheduler_by_name[spec.name]
            if spec.monitor is None:
                try:
                    scheduler.step()
                except TypeError as exc:
                    raise RuntimeError(
                        f"Scheduler '{spec.name}' failed to step without a metric. "
                        "If this scheduler expects a monitored value, configure "
                        f"`schedulers.{spec.name}.monitor`."
                    ) from exc
                continue
            metric = self.trainer.callback_metrics.get(spec.monitor)
            if metric is None:
                raise RuntimeError(
                    f"Scheduler '{spec.name}' expected monitor '{spec.monitor}', "
                    "but that metric was not logged."
                )
            try:
                scheduler.step(metric)
            except TypeError as exc:
                raise RuntimeError(
                    f"Scheduler '{spec.name}' failed to step with monitor "
                    f"'{spec.monitor}'. If this scheduler does not accept a metric, "
                    "remove the monitor setting."
                ) from exc

    def on_save_checkpoint(self, checkpoint: Dict[str, object]) -> None:
        """Persist custom per-optimizer runtime state in checkpoints.

        Lightning already saves and restores the instantiated optimizer and
        scheduler `state_dict()` objects, so this hook only stores the extra
        runtime state introduced by ESPnet3's named multi-optimizer path:

        - `accum_counter`
        - `update_step`

        Scheduler state is not saved here because Lightning's
        checkpoint already contains each scheduler's internal state. Additional
        scheduler fields only need to be added here if ESPnet3 introduces custom
        scheduler-side runtime state that Lightning does not know about.
        """
        if getattr(self.config, "optimizers", None) is not None:
            checkpoint["espnet3_optimizer_runtime_state"] = {
                name: {
                    "accum_counter": state.accum_counter,
                    "update_step": state.update_step,
                }
                for name, state in self._optimizer_states.items()
            }

    def on_load_checkpoint(self, checkpoint: Dict[str, object]) -> None:
        """Restore custom per-optimizer runtime state from checkpoints."""
        runtime_state = checkpoint.get("espnet3_optimizer_runtime_state")
        if not runtime_state:
            return
        self._optimizer_states = {
            name: OptimizerRuntimeState(
                accum_counter=state["accum_counter"],
                update_step=state["update_step"],
            )
            for name, state in runtime_state.items()
        }

    def train_dataloader(self):
        """Build the training DataLoader using ESPnet's DataLoaderBuilder.

        Returns:
            DataLoader: The training DataLoader.
        """
        self.train_dataset.use_espnet_collator = isinstance(
            self.collate_fn, CommonCollateFn
        )
        builder = DataLoaderBuilder(
            dataset=self.train_dataset,
            config=self.config,
            collate_fn=self.collate_fn,
            num_device=self.config.num_device,
            epoch=self.current_epoch,
        )
        with log_stage("train"):
            return builder.build(mode="train")

    def val_dataloader(self):
        """Build the validation DataLoader using ESPnet's DataLoaderBuilder.

        Returns:
            DataLoader: The validation DataLoader.
        """
        self.valid_dataset.use_espnet_collator = isinstance(
            self.collate_fn, CommonCollateFn
        )
        builder = DataLoaderBuilder(
            dataset=self.valid_dataset,
            config=self.config,
            collate_fn=self.collate_fn,
            num_device=self.config.num_device,
            epoch=self.current_epoch,
        )
        with log_stage("valid"):
            return builder.build(mode="valid")

    def state_dict(self, *args, **kwargs):
        """Return the state dict of the model."""
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict into the model."""
        return self.model.load_state_dict(state_dict, strict=strict)

    def collect_stats(self):
        """Collect training and validation statistics using ESPnet's collect_stats.

        Requires `config.stats_dir` to be defined. Saves stats under this directory.

        Raises:
            AssertionError: If `config.stats_dir` is not provided.
        """
        assert hasattr(self.config, "stats_dir"), "config.stats_dir must be defined"

        # Detach dataset/dataloader configs from the root so interpolations like
        # ${dataset_dir} remain resolved when used standalone during collection.
        dataset_config = OmegaConf.create(
            OmegaConf.to_container(self.config.dataset, resolve=True)
        )
        dataloader_config = OmegaConf.create(
            OmegaConf.to_container(self.config.dataloader, resolve=True)
        )

        for mode in ["train", "valid"]:
            if mode == "train":
                dataset_config.preprocessor.train = True
            else:
                dataset_config.preprocessor.train = False

            collect_stats(
                model_config=OmegaConf.to_container(self.config.model, resolve=True),
                dataset_config=dataset_config,
                dataloader_config=dataloader_config,
                mode=mode,
                output_dir=Path(self.config.stats_dir),
                task=getattr(self.config, "task", None),
                parallel_config=(
                    None
                    if "parallel" not in self.config.keys()
                    else self.config.parallel
                ),
                write_collected_feats=False,
            )
