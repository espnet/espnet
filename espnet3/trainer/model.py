"""ESPnet3-specific PyTorch LightningModule wrapper."""

import logging
from pathlib import Path

import lightning
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from espnet2.train.collate_fn import CommonCollateFn
from espnet3.collect_stats import collect_stats
from espnet3.trainer.dataloader import DataLoaderBuilder
from espnet3.trainer.multiple_optim import MultipleOptim
from espnet3.trainer.multiple_scheduler import MultipleScheduler

logger = logging.getLogger("lightning")


class LitESPnetModel(lightning.LightningModule):
    """ESPnet3-specific PyTorch LightningModule wrapper.

    This class handles model training, validation, optimizer/scheduler setup,
    ESPnet-specific dataloader construction, NaN/Inf loss detection across
    distributed setups, and statistics collection.

    Attributes:
        model (torch.nn.Module): The main ESPnet model.
        config (DictConfig): The training configuration.
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
        """Initialize LitESPnetModel."""
        super().__init__()
        self.config = config
        self.model = model
        data_organizer = instantiate(config.dataset)
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
        optim:
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
        optims:
            - optim:
                _target_: torch.optim.Adam
                lr: 0.001
              params: encoder
            - optim:
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
            * Only one of `optim` or `optims` may be specified. Mixing is not allowed.
            * Likewise, `scheduler` and `schedulers` must not be used together.
            * When using `optims`, each `params` must uniquely match a subset of
                trainable parameters.
            * It is an error if:
                * A trainable parameter is assigned to multiple optimizers
                    (overlapping `params`)
                * A trainable parameter is not assigned to any optimizer
                * Any optimizer block is missing `params` or nested `optim`

        Returns:
            dict: A dictionary with keys `"optimizer"` and `"lr_scheduler"`
                for PyTorch Lightning.

        Raises:
            AssertionError: If configuration rules are violated.
            ValueError: If neither optimizer configuration is provided.
        """
        if getattr(self.config, "optim", None) and getattr(
            self.config, "scheduler", None
        ):
            # setup optimizer and scheduler
            assert (
                getattr(self.config, "optims", None) is None
            ), "Mixture of `optim` and `optims` is not allowed."
            params = filter(lambda p: p.requires_grad, self.parameters())
            optimizer = instantiate(
                OmegaConf.to_container(self.config.optim, resolve=True), params
            )

            assert (
                getattr(self.config, "schedulers", None) is None
            ), "Mixture of `scheduler` and `schedulers` is not allowed."
            scheduler = instantiate(
                OmegaConf.to_container(self.config.scheduler, resolve=True),
                optimizer=optimizer,
            )

        elif getattr(self.config, "optims", None) and getattr(
            self.config, "schedulers", None
        ):
            assert (
                getattr(self.config, "optim", None) is None
            ), "Mixture of `optim` and `optims` is not allowed."
            assert len(self.config.optims) == len(self.config.schedulers), (
                "The number of optimizers and schedulers must be equal: "
                + f"optims: {len(self.config.optims)}, "
                + f"schedulers: {len(self.config.schedulers)}"
            )

            optims = []
            trainable_params = {
                name: param
                for name, param in self.named_parameters()
                if param.requires_grad
            }  # key: name, value: param
            used_param_ids = set()

            for optim_cfg in self.config.optims:
                assert "params" in optim_cfg, "missing 'params' in optim config"
                assert "optim" in optim_cfg, "missing nested 'optim' block"

                # filter parameters whose name contains the 'params' keyword
                selected = [
                    p
                    for name, p in trainable_params.items()
                    if optim_cfg["params"] in name
                ]
                selected_names = [
                    name
                    for name in trainable_params.keys()
                    if optim_cfg["params"] in name
                ]
                assert (
                    len(selected) > 0
                ), f"No trainable parameters found for: {optim_cfg['params']}"

                for n in selected_names:
                    assert (
                        n not in used_param_ids
                    ), f"Parameter {n} is assigned to multiple optimizers"
                    used_param_ids.add(n)

                optim = instantiate(
                    OmegaConf.to_container(optim_cfg["optim"], resolve=True), selected
                )
                optims.append(optim)

            # Check for uncovered parameters
            all_param_ids = {
                name for name, p in self.named_parameters() if p.requires_grad
            }
            unused_param_ids = all_param_ids - used_param_ids
            assert (
                not unused_param_ids
            ), f"{unused_param_ids} are not assigned to any optimizer"

            optimizer = MultipleOptim(optims)

            assert (
                getattr(self.config, "scheduler", None) is None
            ), "Mixture of `scheduler` and `schedulers` is not allowed."
            schedulers = []
            for i_sch, scheduler in enumerate(self.config.schedulers):
                schedulers.append(
                    instantiate(
                        OmegaConf.to_container(scheduler.scheduler, resolve=True),
                        optimizer=optims[i_sch],
                    )
                )

            scheduler = [
                MultipleScheduler(optimizer, sch, i_sch)
                for i_sch, sch in enumerate(schedulers)
            ]
        else:
            raise ValueError(
                "Must specify either `optim` or `optims` and `scheduler` or"
                "`schedulers`"
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # assuming lr scheduler is updated per step
            },
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
        return builder.build(mode="valid")

    def state_dict(self, *args, **kwargs):
        """Return the state dict of the model."""
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        """Load state dict into the model."""
        return self.model.load_state_dict(state_dict, strict=strict)

    def collect_stats(self):
        """Collect training and validation statistics using ESPnet's collect_stats.

        Requires `config.statsdir` to be defined. Saves stats under this directory.

        Raises:
            AssertionError: If `config.statsdir` is not provided.
        """
        assert hasattr(self.config, "statsdir"), "config.statsdir must be defined"

        for mode in ["train", "valid"]:
            collect_stats(
                model_config=OmegaConf.to_container(self.config.model, resolve=True),
                dataset_config=self.config.dataset,
                dataloader_config=self.config.dataloader,
                mode=mode,
                output_dir=Path(self.config.statsdir),
                task=getattr(self.config, "task", None),
                parallel_config=(
                    None
                    if "parallel" not in self.config.keys()
                    else self.config.parallel
                ),
                write_collected_feats=False,
            )
