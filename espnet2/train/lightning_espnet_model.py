from pathlib import Path

import lightning as L
import torch

from espnet2.tasks.abs_task import optim_classes, scheduler_classes
from espnet2.tasks.asr import ASRTask
from espnet2.train.distributed_utils import DistributedOption
from espnet2.utils.yaml_no_alias_safe_dump import yaml_no_alias_safe_dump

task_choices = {
    "asr": ASRTask,
}


class LitESPnetModel(L.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()  # args now in self.hparams
        self.args = args
        self.task_class = task_choices[args.task]
        self.model = self.task_class.build_model(args=args)

        # Save config to make it compatible with ESPnet inference
        if self.global_rank == 0:
            with (Path(args.output_dir) / "config.yaml").open(
                "w", encoding="utf-8"
            ) as f:
                yaml_no_alias_safe_dump(vars(args), f, indent=4, sort_keys=False)

    def _step(self, batch, batch_idx, mode):
        utt_id, batch = batch
        batch["utt_id"] = utt_id

        # loss is averaged over samples within a mini-batch; weight is batch size
        loss, stats, weight = self.model(**batch)

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
                "interval": "step",  # assuming lr scheduler is updated per step (not epoch)
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
