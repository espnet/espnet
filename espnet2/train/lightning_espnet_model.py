import logging
from pathlib import Path

import lightning as L
import torch
import torch.distributed

from espnet2.tasks.abs_task import optim_classes, scheduler_classes
from espnet2.tasks.asr import ASRTask
from espnet2.tasks.cls import CLSTask
from espnet2.train.distributed_utils import DistributedOption
from espnet2.utils.yaml_no_alias_safe_dump import yaml_no_alias_safe_dump

logger = logging.getLogger("lightning")

task_choices = {
    "asr": ASRTask,
    "cls": CLSTask,
}


class LitESPnetModel(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()  # args now in self.hparams
        self.args = args
        self.task_class = task_choices[args.task]
        self.model = self.task_class.build_model(args=args)

        if self.global_rank == 0:
            # Save config to make it compatible with ESPnet inference
            with (Path(args.output_dir) / "config.yaml").open(
                "w", encoding="utf-8"
            ) as f:
                yaml_no_alias_safe_dump(vars(args), f, indent=4, sort_keys=False)

    def _sync2skip(self, flag_skip):
        # see below:
        # https://github.com/Lightning-AI/lightning/issues/5243#issuecomment-1552650013
        # gathering a tensor across all workers and then reduce it using or
        world_size = torch.distributed.get_world_size()
        torch.distributed.barrier()
        # now gather
        result = [torch.zeros_like(flag_skip) for _ in range(world_size)]
        torch.distributed.all_gather(result, flag_skip)
        any_invalid = torch.sum(torch.stack(result)).bool().item()
        return any_invalid

    def _check_nan_inf_loss(self, loss, batch_id):
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
        utt_id, batch = batch
        batch["utt_id"] = utt_id

        # loss is averaged over samples within a mini-batch; weight is batch size
        loss, stats, weight = self.model(**batch)

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
        return self._step(batch, batch_idx, mode="train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, batch_idx, mode="valid")

    def configure_optimizers(self):
        optim_class = optim_classes.get(self.args.optim)
        if optim_class is None:
            raise ValueError(f"must be one of {list(optim_classes)}: {self.args.optim}")
        optimizer = optim_class(
            filter(lambda p: p.requires_grad, self.parameters()), **self.args.optim_conf
        )

        name = self.args.scheduler
        conf = self.args.scheduler_conf
        if name is not None:
            cls_ = scheduler_classes.get(name)
            if cls_ is None:
                raise ValueError(f"must be one of {list(scheduler_classes)}: {name}")
            scheduler = cls_(optimizer, **conf)
        else:
            scheduler = None

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                # assuming lr scheduler is updated per step (not epoch)
                "interval": "step",
            },
        }

    def train_dataloader(self):
        if self.args.multiple_iterator:
            train_iter_factory = self.task_class.build_multiple_iter_factory(
                args=self.args,
                distributed_option=DistributedOption(distributed=True),
                mode="train",
            )
        else:
            train_iter_factory = self.task_class.build_iter_factory(
                args=self.args,
                distributed_option=DistributedOption(distributed=True),
                mode="train",
            )
        return train_iter_factory.build_iter(epoch=self.current_epoch)

    def val_dataloader(self):
        valid_iter_factory = self.task_class.build_iter_factory(
            args=self.args,
            distributed_option=DistributedOption(distributed=True),
            mode="valid",
        )
        return valid_iter_factory.build_iter(epoch=self.current_epoch)
