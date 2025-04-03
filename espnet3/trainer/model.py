import os
from pathlib import Path

import lightning as L
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

from espnet2.train.collate_fn import CommonCollateFn
from espnet3.trainer.hybrid_optim import HybridOptim
from espnet3.trainer.hybrid_scheduler import HybridLRS


class LitESPnetModel(L.LightningModule):
    def __init__(self, model, config, train_dataset, valid_dataset):
        super().__init__()
        self.config = config
        self.model = model
        self.train_dataset = train_dataset
        self.valid_dataset = valid_dataset
        self.save_hyperparameters()  # args now in self.hparams

        # Save config to make it compatible with ESPnet inference
        if self.global_rank == 0:
            if not os.path.exists(Path(self.config.expdir)):
                os.makedirs(Path(self.config.expdir))

            with (Path(self.config.expdir) / "config.yaml").open(
                "w", encoding="utf-8"
            ) as f:
                f.write(yaml.dump(vars(self.config)))

    def _step(self, batch, batch_idx, mode):
        # loss is averaged over samples within a mini-batch; weight is batch size
        loss, stats, weight = self.model(**batch[1])

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
            # batch_size=weight.item(),
        )
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="valid")

    def configure_optimizers(self):
        """
        This implementation utilize the following document to use multiple
        optimizers and schedulers to the different modules:
        https://github.com/Lightning-AI/pytorch-lightning/issues/3346
        If you want to use multiple optimizers or schedulers to the different modules,
        See <espnet-ez document> for how to do it.

        Default to one optimizer and scheduler.
        Default optimizer is Adam and scheduler is torch.optim.lr_scheduler.StepLR
        with default values in PyTorch.
        """
        if getattr(self.config, "optim") and getattr(self.config, "scheduler"):
            # setup optimizer and scheduler
            assert (
                getattr(self.config, "optims", None) is None
            ), "Mixture of `optim` and `optims` is not allowed."
            params = filter(lambda p: p.requires_grad, self.parameters())
            optimizer = instantiate(self.config.optim, params=params)

            assert (
                getattr(self.config, "schedulers", None) is None
            ), "Mixture of `scheduler` and `schedulers` is not allowed."
            scheduler = instantiate(self.config.scheduler, optimizer=optimizer)

        elif getattr(self.config, "optims") and getattr(self.config, "schedulers"):
            assert (
                getattr(self.config, "optim", None) is None
            ), "Mixture of `optim` and `optims` is not allowed."
            assert len(self.config.optims) == len(self.config.schedulers), (
                f"The number of optimizers and schedulers must be equal: "
                + f"optims: {len(self.config.optims)}, "
                + f"schedulers: {len(self.config.schedulers)}"
            )
            optims = []
            trainable_params = filter(lambda p: p.requires_grad, self.parameters())
            for optim in self.config.optims:
                # filter params for optimizer
                assert "params" in optim, "missing 'params' in optim config"
                params = [p for p in trainable_params if optim["params"] in p.name]
                assert len(params) > 0, (
                    f"No trainable parameters found for"
                    + f"optimizer: {optim} with params: {optim['params']}"
                )
                optims.append(instantiate(optim, params=params))
            optimizer = HybridOptim(optims)

            assert (
                getattr(self.config, "scheduler", None) is None
            ), "Mixture of `scheduler` and `schedulers` is not allowed."
            schedulers = []
            for i_sch, scheduler in enumerate(self.config.schedulers):
                schedulers.append(instantiate(scheduler, optimizer=optims[i_sch]))
            scheduler = [
                HybridLRS(optimizer, i_sch, sch) for i_sch, sch in enumerate(schedulers)
            ]
        elif not getattr(self.config, "optim") and not getattr(
            self.config, "scheduler"
        ):
            optimizer = torch.optim.Adam(self.parameters())
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer)
        else:
            raise ValueError(
                "Must specify either `optim` or `optims` and `scheduler` or `schedulers`"
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",  # assuming lr scheduler is updated per step (not epoch)
            },
        }

    def train_dataloader(self):
        # Define sampler if specified
        dataloader_config = {}
        sampler = None
        train_dataset = self.train_dataset
        collate_fn = CommonCollateFn(int_pad_value=-1)
        if hasattr(self.config, "dataloader"):
            if hasattr(self.config.dataloader, "train"):
                dataloader_config.update(
                    OmegaConf.to_container(self.config.dataloader.train)
                )

            if hasattr(self.config.dataloader, "train") and hasattr(
                self.config.dataloader.train, "sampler"
            ):
                sampler = instantiate(
                    self.config.dataloader.train.sampler, self.train_dataset
                )

            # define collate_fn. Default to ESPnet's CommonCollateFn
            if hasattr(self.config.dataloader, "collate_fn"):
                collate_fn = instantiate(self.config.dataloader.collate_fn)

            if "sampler" in dataloader_config:
                dataloader_config.pop("sampler")

            if (
                "dataset" in dataloader_config
            ):  # remove dataset for compatibility with torch.utils.data.DataLoader
                dataloader_config.pop("dataset")

        return torch.utils.data.DataLoader(
            train_dataset, sampler=sampler, collate_fn=collate_fn, **dataloader_config
        )

    def val_dataloader(self):
        # Define sampler if specified
        dataloader_config = {}
        sampler = None
        valid_dataset = self.valid_dataset
        collate_fn = CommonCollateFn(int_pad_value=-1)
        if hasattr(self.config, "dataloader"):
            if hasattr(self.config.dataloader, "valid"):
                dataloader_config.update(
                    OmegaConf.to_container(self.config.dataloader.valid)
                )

            if hasattr(self.config.dataloader, "valid") and hasattr(
                self.config.dataloader.valid, "sampler"
            ):
                sampler = instantiate(
                    self.config.dataloader.valid.sampler, self.valid_dataset
                )

            # define collate_fn. Default to ESPnet's CommonCollateFn
            if hasattr(self.config.dataloader, "collate_fn"):
                collate_fn = instantiate(self.config.dataloader.collate_fn)

            if "sampler" in dataloader_config:
                dataloader_config.pop("sampler")

            if (
                "dataset" in dataloader_config
            ):  # remove dataset for compatibility with torch.utils.data.DataLoader
                dataloader_config.pop("dataset")

        return torch.utils.data.DataLoader(
            valid_dataset, sampler=sampler, collate_fn=collate_fn, **dataloader_config
        )

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True):
        return self.model.load_state_dict(state_dict, strict=strict)
