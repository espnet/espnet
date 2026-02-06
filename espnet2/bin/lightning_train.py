import argparse
import copy
import importlib
import logging
from pathlib import Path

import lightning as L
import torch
from lightning.pytorch.callbacks import (
    LearningRateMonitor,
    ModelCheckpoint,
    TQDMProgressBar,
)
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from lightning.pytorch.strategies import DDPStrategy, FSDPStrategy
from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

from espnet2.train.lightning_callbacks import (
    AverageCheckpointsCallback,
    user_callback_choices,
)
from espnet2.train.lightning_espnet_model import LitESPnetModel, task_choices
from espnet2.utils.nested_dict_action import NestedDictAction


def get_base_parser():
    """Create the base parser with task selection."""
    parser = argparse.ArgumentParser(
        description="Launch training using Lightning AI backend. ",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=list(task_choices.keys()),
        help="Task to execute.",
    )
    return parser


def get_parser(task_class=None):
    if task_class is None:
        parser = get_base_parser()
    else:
        parser = task_class.get_parser()
    parser.add_argument(
        "--lightning_conf",
        action=NestedDictAction,
        default=dict(),
        help="Arguments related to Lightning Trainer.",
    )
    parser.add_argument(
        "--user_callbacks",
        type=str,
        nargs="+",
        choices=list(user_callback_choices.keys()),
        default=[],
        help="User-defined callbacks.",
    )
    return parser


def build_user_callbacks(user_callbacks):
    callbacks = []
    for callback_name in user_callbacks:
        callback_cls = user_callback_choices[callback_name]
        callbacks.append(callback_cls())
    return callbacks


def main():
    # First parse the task and then parse task-specific arguments
    base_parser = get_base_parser()
    task_args, remaining_args = base_parser.parse_known_args()
    task_class = task_choices[task_args.task]
    parser = get_parser(task_class)
    args = parser.parse_args(remaining_args)
    args.task = task_args.task

    # Set logging level
    logging.getLogger("lightning").setLevel(getattr(logging, args.log_level))

    # Set random seed
    L.seed_everything(args.seed)

    # Set additional configurations that might be helpful
    torch.set_float32_matmul_precision("high")

    # Instantiate the Lightning Model
    lit_model = LitESPnetModel(args=args)

    # Instantiate the strategy
    trainer_conf = copy.deepcopy(args.lightning_conf)
    strategy = trainer_conf.pop("strategy", "ddp")
    strategy_conf = trainer_conf.pop("strategy_conf", dict())

    if strategy == "ddp":
        ddp_comm_hook = strategy_conf.pop("ddp_comm_hook", None)
        if ddp_comm_hook is not None:
            ddp_comm_hook = getattr(default_hooks, ddp_comm_hook)

        strategy = DDPStrategy(
            ddp_comm_hook=ddp_comm_hook,
            **strategy_conf,
        )

    elif strategy == "fsdp":
        auto_wrap_policy = strategy_conf.pop("auto_wrap_policy", None)
        if auto_wrap_policy is not None and len(auto_wrap_policy) > 0:
            auto_wrap_policy = set(
                getattr(
                    importlib.import_module(".".join(policy.split(".")[:-1])),
                    policy.split(".")[-1],
                )
                for policy in auto_wrap_policy
            )
        else:
            auto_wrap_policy = None

        activation_checkpointing_policy = strategy_conf.pop(
            "activation_checkpointing_policy", None
        )
        if (
            activation_checkpointing_policy is not None
            and len(activation_checkpointing_policy) > 0
        ):
            activation_checkpointing_policy = set(
                getattr(
                    importlib.import_module(".".join(policy.split(".")[:-1])),
                    policy.split(".")[-1],
                )
                for policy in activation_checkpointing_policy
            )
        else:
            activation_checkpointing_policy = None

        strategy = FSDPStrategy(
            auto_wrap_policy=auto_wrap_policy,
            activation_checkpointing_policy=activation_checkpointing_policy,
            **strategy_conf,
        )

    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    # Create callbacks
    # Save full last checkpoint to resume training
    last_ckpt_callback = ModelCheckpoint(
        dirpath=args.output_dir,
        save_last="link",
        filename="step{step}",
        auto_insert_metric_name=False,
        save_on_train_epoch_end=True,
        save_weights_only=False,
    )

    # Save best models (weights only)
    best_ckpt_callbacks = []
    for monitor, mode, nbest in trainer_conf.pop("best_model_criterion", []):
        best_ckpt_callbacks.append(
            ModelCheckpoint(
                save_top_k=nbest,
                monitor=monitor,
                mode=mode,  # "min" or "max"
                dirpath=args.output_dir,
                save_last=False,
                # Add monitor to filename to avoid overwriting
                # when multiple metrics are used
                filename="epoch{epoch}_step{step}_" + monitor.replace("/", "."),
                auto_insert_metric_name=False,
                save_on_train_epoch_end=False,
                save_weights_only=True,
                enable_version_counter=False,  # just overwrite
            )
        )

    # Average best models after training
    ave_ckpt_callback = AverageCheckpointsCallback(
        output_dir=args.output_dir, best_ckpt_callbacks=best_ckpt_callbacks
    )

    # Get user callbacks
    user_callbacks = build_user_callbacks(args.user_callbacks)

    # Monitor learning rate
    lr_callback = LearningRateMonitor()

    # Create loggers
    loggers = []

    if args.use_tensorboard:
        tb_logger = TensorBoardLogger(
            save_dir=args.output_dir,
            name="lightning_logs",
        )
        loggers.append(tb_logger)

    if args.use_wandb:
        wandb_latest_id = None
        # Resume the latest run if exists by setting the version
        if (Path(args.output_dir) / "wandb" / "latest-run").exists():
            wandb_latest_id = (
                (Path(args.output_dir) / "wandb" / "latest-run")
                .resolve()
                .name.split("-")[-1]
            )
        wandb_logger = WandbLogger(
            project=args.wandb_project or "ESPnet_" + task_class.__name__,
            name=args.wandb_name or str(Path(".").resolve()).replace("/", "_"),
            save_dir=args.output_dir,
            version=wandb_latest_id,
        )
        loggers.append(wandb_logger)

    # Instantiate the Lightning Trainer
    trainer = L.Trainer(
        # Reload dataloaders every epoch to reuse ESPnet's dataloader
        reload_dataloaders_every_n_epochs=1,
        # ESPnet's dataloader already shards the dataset based on distributed setups
        use_distributed_sampler=False,
        **trainer_conf,
        callbacks=[
            last_ckpt_callback,
            *best_ckpt_callbacks,
            ave_ckpt_callback,
            lr_callback,
            TQDMProgressBar(refresh_rate=args.lightning_conf["log_every_n_steps"]),
        ]
        + user_callbacks,
        strategy=strategy,
        logger=loggers,
    )

    # Start training with automatic resuming from the last checkpoint
    trainer.fit(model=lit_model, ckpt_path="last")


if __name__ == "__main__":
    main()
