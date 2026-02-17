"""ESPnet3 PyTorch LightningModule for training and data integration."""

import logging
from pathlib import Path

import lightning
import torch
from humanfriendly import format_number, format_size
from hydra.utils import instantiate
from omegaconf import OmegaConf

from espnet2.train.collate_fn import CommonCollateFn
from espnet3.components.data.collect_stats import collect_stats
from espnet3.components.data.dataloader import DataLoaderBuilder
from espnet3.components.optimizers.multiple_optimizer import MultipleOptimizer
from espnet3.components.optimizers.multiple_scheduler import MultipleScheduler
from espnet3.utils.logging_utils import log_component, log_stage

logger = logging.getLogger("lightning")


class ESPnetLightningModule(lightning.LightningModule):
    """ESPnet3 LightningModule wrapper for model training and data integration.

    This module follows Lightning best practices by defining the training/validation
    steps, optimizer/scheduler configuration, and dataloader hooks on the
    LightningModule itself. It also integrates ESPnet-specific dataset handling,
    distributed NaN/Inf loss skipping, and statistics collection.

    Attributes:
        model (torch.nn.Module): The main ESPnet model.
        config (DictConfig): Training configuration for model and data setup.
        train_dataset: Training dataset organized by the ESPnet data organizer.
        valid_dataset: Validation dataset.
        collate_fn (Callable): Collation function used in DataLoader.
        is_espnet_sampler (bool): Whether the model uses ESPnet's custom sampler.

    Note:
        This class assumes the use of a `DataOrganizer`-compatible dataset config.
        The `data_organizer` is instantiated temporarily to access
        `train` and `valid` datasets, but is not retained as an attribute
        since it is no longer needed after extraction.
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

        # If user is trying to use both Pytorch dataloader and ESPnet's dataloader
        # Then raise an error here.
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
        # define collate_fn. Default to ESPnet's CommonCollateFn
        if hasattr(self.config.dataloader, "collate_fn"):
            self.collate_fn = instantiate(self.config.dataloader.collate_fn)

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

    def _check_nan_inf_loss(self, loss, batch_id):
        """Check for NaN or Inf in loss and synchronize across all workers.

        Args:
            loss (torch.Tensor): The computed loss tensor.
            batch_id (int): Batch index (for logging).

        Returns:
            bool: True if any worker observed an invalid loss and requested a skip.
        """
        mask_nan_inf = torch.logical_or(torch.isnan(loss), ~torch.isfinite(loss))
        if torch.any(mask_nan_inf):
            # if any is invalid then we must flag this to all DDP processes
            flag_skip = torch.ones((), device=loss.device, dtype=torch.bool)
        else:
            flag_skip = torch.zeros((), device=loss.device, dtype=torch.bool)

        # sub-optimal but will do, till they fix it in
        # https://github.com/Lightning-AI/lightning/issues/5243#issuecomment-1552650013
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
            # reset counter
            self.nan_countdown = 1

        return any_invalid

    def _step(self, batch, batch_idx, mode):
        """Shared logic for training and validation steps.

        Args:
            batch (Tuple): Tuple of (id, inputs), passed to model.
            batch_idx (int): Batch index.
            mode (str): Either "train" or "valid".

        Returns:
            Optional[torch.Tensor]: Loss value if valid, None if skipped due to NaN.
        """
        # loss is averaged over samples within a mini-batch; weight is batch size
        loss, stats, weight = self.model(**batch[1])

        any_invalid = self._check_nan_inf_loss(loss, batch_idx)
        if any_invalid:
            # skip this batch altogether on all workers.
            return None

        new_stats = {}
        for k, v in stats.items():
            if v is not None:
                new_stats[f"{mode}/{k}"] = v.item()

        self.log_dict(
            new_stats,
            prog_bar=True,
            logger=True,
            sync_dist=(mode == "valid"),
            # NOTE(Yifan): must convert weight to a number to avoid device mismatch
            # when resuming training from a checkpoint with checkpoint callbacks
            batch_size=weight.item(),
        )
        return loss

    def training_step(self, batch, batch_idx):
        """Training step logic."""
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        """Validate step."""
        return self._step(batch, batch_idx, mode="valid")

    def configure_optimizers(self):
        """Configure optimizers and schedulers for training.

        This method supports two modes of configuration.

        1. Single Optimizer + Scheduler:
        Use when the entire model is trained with a single optimizer.
        ```yaml
        optimizer:
            _target_: torch.optim.Adam
            lr: 0.001

        scheduler:
            _target_: torch.optim.lr_scheduler.StepLR
            step_size: 10
        ````

        2. Multiple Optimizers + Schedulers:
        Use when training different parts of the model with different optimizers.
        Each optimizer block must contain both a nested `optim` config and a `params`
        key indicating a substring to match parameter names.

        ```yaml
        optimizers:
            - optimizer:
                _target_: torch.optim.Adam
                lr: 0.001
              params: encoder
            - optimizer:
                _target_: torch.optim.SGD
                lr: 0.01
              params: decoder

        schedulers:
            - scheduler:
                _target_: torch.optim.lr_scheduler.StepLR
                step_size: 10
            - scheduler:
                _target_: torch.optim.lr_scheduler.ReduceLROnPlateau
                patience: 2
        ```

        Notes:
            * Only one of `optimizer` or `optimizers` may be specified.
                Mixing is not allowed.
            * Likewise, `scheduler` and `schedulers` must not be used together.
            * When using `optimizers`, each `params` must uniquely match a subset of
                trainable parameters.
            * It is an error if:
                * A trainable parameter is assigned to multiple optimizers
                    (overlapping `params`)
                * A trainable parameter is not assigned to any optimizer
                * Any optimizer block is missing `params` or nested `optimizer`

        Returns:
            dict: A dictionary with keys `"optimizer"` and `"lr_scheduler"`
                for PyTorch Lightning.

        Raises:
            AssertionError: If configuration rules are violated.
            ValueError: If neither optimizer configuration is provided.
        """
        if getattr(self.config, "optimizer", None) and getattr(
            self.config, "scheduler", None
        ):
            # setup optimizer and scheduler
            assert (
                getattr(self.config, "optimizers", None) is None
            ), "Mixture of `optimizer` and `optimizers` is not allowed."
            params = filter(lambda p: p.requires_grad, self.parameters())
            optimizer = instantiate(
                OmegaConf.to_container(self.config.optimizer, resolve=True), params
            )

            assert (
                getattr(self.config, "schedulers", None) is None
            ), "Mixture of `scheduler` and `schedulers` is not allowed."
            scheduler = instantiate(
                OmegaConf.to_container(self.config.scheduler, resolve=True),
                optimizer=optimizer,
            )

        elif getattr(self.config, "optimizers", None) and getattr(
            self.config, "schedulers", None
        ):
            assert (
                getattr(self.config, "optimizer", None) is None
            ), "Mixture of `optimizer` and `optimizers` is not allowed."
            assert len(self.config.optimizers) == len(self.config.schedulers), (
                "The number of optimizers and schedulers must be equal: "
                + f"optimizers: {len(self.config.optimizers)}, "
                + f"schedulers: {len(self.config.schedulers)}"
            )

            optimizers = []
            trainable_params = {
                name: param
                for name, param in self.named_parameters()
                if param.requires_grad
            }  # key: name, value: param
            used_param_ids = set()

            for optim_config in self.config.optimizers:
                assert "params" in optim_config, "missing 'params' in optimizer config"
                assert "optimizer" in optim_config, "missing nested 'optimizer' block"

                # filter parameters whose name contains the 'params' keyword
                selected = [
                    p
                    for name, p in trainable_params.items()
                    if optim_config["params"] in name
                ]
                selected_names = [
                    name
                    for name in trainable_params.keys()
                    if optim_config["params"] in name
                ]
                assert (
                    len(selected) > 0
                ), f"No trainable parameters found for: {optim_config['params']}"

                for n in selected_names:
                    assert (
                        n not in used_param_ids
                    ), f"Parameter {n} is assigned to multiple optimizers"
                    used_param_ids.add(n)

                optimizer = instantiate(
                    OmegaConf.to_container(optim_config["optimizer"], resolve=True),
                    selected,
                )
                optimizers.append(optimizer)

            # Check for uncovered parameters
            all_param_ids = {
                name for name, p in self.named_parameters() if p.requires_grad
            }
            unused_param_ids = all_param_ids - used_param_ids
            assert (
                not unused_param_ids
            ), f"{unused_param_ids} are not assigned to any optimizer"

            optimizer = MultipleOptimizer(optimizers)

            assert (
                getattr(self.config, "scheduler", None) is None
            ), "Mixture of `scheduler` and `schedulers` is not allowed."
            schedulers = []
            for i_sch, scheduler in enumerate(self.config.schedulers):
                schedulers.append(
                    instantiate(
                        OmegaConf.to_container(scheduler.scheduler, resolve=True),
                        optimizer=optimizers[i_sch],
                    )
                )

            scheduler = [
                MultipleScheduler(optimizer, sch, i_sch)
                for i_sch, sch in enumerate(schedulers)
            ]
        else:
            raise ValueError(
                "Must specify either `optimizer` or `optimizers` and `scheduler` or"
                "`schedulers`"
            )

        self._log_training_summary(
            logger,
            self.model,
            optimizer=optimizer,
            scheduler=scheduler,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # assuming lr scheduler is updated per step
            },
        }

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
            the training loop is configured.

        Args:
            logger (logging.Logger): Logger used to emit messages.
            model: PyTorch model to summarize.
            optimizer: Optimizer instance or wrapper.
            scheduler: Scheduler instance or list of schedulers.

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
        logger.log(
            logging.INFO,
            "    DType composition: %s",
            dtype_desc,
            stacklevel=2,
        )

        if optimizer is None and scheduler is None:
            return

        if isinstance(optimizer, MultipleOptimizer):
            optimizers = list(optimizer.optimizers)
        elif isinstance(optimizer, list):
            optimizers = optimizer
        else:
            optimizers = [optimizer]
        for idx, optim in enumerate(optimizers):
            logger.log(logging.INFO, "Optimizer[%d]:", idx, stacklevel=2)
            log_component(
                logger,
                kind="Optimizer",
                label=str(idx),
                obj=optim,
                max_depth=2,
            )

        if scheduler is None:
            return
        if isinstance(scheduler, list):
            schedulers = scheduler
        else:
            schedulers = [scheduler]
        for idx, sch in enumerate(schedulers):
            logger.log(logging.INFO, "Scheduler[%d]:", idx, stacklevel=2)
            log_component(
                logger,
                kind="Scheduler",
                label=str(idx),
                obj=sch,
                max_depth=2,
            )

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
