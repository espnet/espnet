# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""TorchTitan-based trainer implementation for SpeechLM training.

This trainer uses FSDP2 for data parallelism, providing an alternative to
DeepSpeed-based training. It maintains interface compatibility with DeepSpeedTrainer.
"""

import copy
import gc
import itertools
import logging
import math
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import wandb
from torch.distributed.checkpoint.default_planner import DefaultLoadPlanner
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torchtitan.distributed import utils as dist_utils

from espnet2.speechlm.model.speechlm.parallel_utils import (
    init_parallel_dims,
    parallel_strategies,
)
from espnet2.speechlm.utils.data import to_device
from espnet2.speechlm.utils.model_summary import model_summary

logger = logging.getLogger(__name__)


def reinit_model(model, trainer_args):
    """Re-initialize all model parameters for training from scratch.

    Norm weights → 1.0, biases → 0.0, everything else → N(0, std).
    Skipped entirely when model_init is not set in config.
    """
    if trainer_args.get("model_init", None) != "normal":
        return

    std = float(trainer_args.get("model_init_std", 0.02))

    # Verify all ranks share the same RNG state before re-init.
    if dist.is_initialized():
        local_hash = torch.tensor(
            [torch.initial_seed() % (2**31)], dtype=torch.long, device="cuda"
        )
        gathered = [torch.zeros_like(local_hash) for _ in range(dist.get_world_size())]
        dist.all_gather(gathered, local_hash)
        seeds = [t.item() for t in gathered]
        if len(set(seeds)) != 1:
            raise RuntimeError(
                f"RNG state mismatch across ranks before reinit_model! "
                f"Seeds: {seeds}. This would cause HSDP replicas to diverge."
            )

    logging.info(
        f"Re-initializing all parameters: normal(std={std}), norm=1.0, bias=0.0"
    )
    for name, p in model.named_parameters():
        if "norm" in name or "layernorm" in name:
            nn.init.ones_(p)
        elif "bias" in name:
            nn.init.zeros_(p)
        else:
            nn.init.normal_(p, mean=0.0, std=std)


class TitanTrainer:
    """TorchTitan-based trainer with FSDP2 support for SpeechLM.

    This trainer provides distributed training using PyTorch's native FSDP2
    (Fully Sharded Data Parallel) instead of DeepSpeed ZeRO. It maintains
    interface compatibility with DeepSpeedTrainer for easy switching.

    IMPORTANT: wandb is MANDATORY and must be initialized before creating this trainer.
    The trainer will raise an error if wandb.run is None.
    Wandb should always be initialized in offline mode for local-only logging.
    All training metrics, losses, and stats are logged to local wandb files.

    Key Features:
        - FSDP2/HSDP for memory-efficient data parallelism
        - Activation checkpointing for memory optimization
        - PyTorch Distributed Checkpoint (DCP) for reshardable checkpoints
        - Compatible with existing DataIteratorFactory interface
    """

    count_normalized_keys = [
        "loss",
        "ce_loss",
        "z_loss",
        "z_loss_s0",
        "z_loss_mm",
        "load_balance_loss",
        "acc_layer0",
    ]

    def __init__(
        self,
        train_data_factory,
        valid_data_factories: Dict,
        model: nn.Module,
        resume_path: Optional[Path],
        output_dir: Path,
        trainer_args: Dict[str, Any],
        parallel_dims=None,
    ):
        """Initialize TorchTitan trainer.

        Args:
            train_data_factory: Training data iterator factory
            valid_data_factories: Dictionary of validation data factories
            model: Model to train (HuggingFace model)
            resume_path: Path to checkpoint for resuming training
            output_dir: Directory for saving outputs
            trainer_args: Training configuration dictionary containing:
                - max_step: Maximum number of training steps
                - log_interval: Steps between logging
                - save_interval: Steps between checkpoints
                - freeze_param: List of parameter prefixes to freeze
                - titan_config: TorchTitan configuration dict with:
                    - dp_shard: FSDP degree (-1 = auto)
                    - dp_replicate: HSDP replicate degree (default: 1)
                    - mixed_precision_param: Parameter dtype (default: "bfloat16")
                    - mixed_precision_reduce: Reduce dtype (default: "float32")
                    - gradient_clipping: Max gradient norm (default: 1.0)
                    - optimizer: Optimizer config dict
                    - lr_scheduler: LR scheduler config dict
                    - activation_checkpoint: "none", "selective", or "full"
            parallel_dims: Pre-built ParallelDims from train.py (avoids
                double init_parallel_dims). If None, builds internally.
        """
        if wandb.run is None:
            raise RuntimeError(
                "wandb must be initialized before creating TitanTrainer. "
                "Please call wandb.init() first."
            )

        self.train_data_factory = train_data_factory
        self.valid_data_factories = valid_data_factories
        self.output_dir = Path(output_dir)
        self.trainer_args = trainer_args
        (self.output_dir / "checkpoints").mkdir(exist_ok=True, parents=True)

        self.global_step = 0
        self.max_step = trainer_args["max_step"]
        self.save_interval = trainer_args["save_interval"]
        self.log_interval = trainer_args["log_interval"]
        self.gradient_accumulation_steps = trainer_args.get(
            "gradient_accumulation_steps", 1
        )

        # Freeze parameters
        for t in trainer_args.get("freeze_param", []):
            if isinstance(model, torch.nn.ModuleList):
                param_iter = itertools.chain.from_iterable(
                    m.named_parameters() for m in model
                )
            else:
                param_iter = model.named_parameters()
            for k, p in param_iter:
                if k.startswith(t + ".") or k == t:
                    logger.info(f"Setting {k}.requires_grad = False")
                    p.requires_grad = False

        # Initialize distributed environment and ParallelDims
        titan_config = trainer_args.get("titan_config", {})

        # Determine training dtype and cast model before FSDP wrapping
        # This ensures all parameters have uniform dtype for FSDP
        dtype_str = titan_config.get("mixed_precision_param", "bfloat16")
        self.dtype = getattr(torch, dtype_str)
        storage_dtype_str = titan_config.get("optimizer_storage_dtype", "float32")
        self.storage_dtype = getattr(torch, storage_dtype_str)
        self.max_norm = titan_config.get("gradient_clipping", 1.0)

        if parallel_dims is not None:
            self.parallel_dims = parallel_dims
            self.local_rank = torch.cuda.current_device()
            self.global_rank = dist.get_rank()
        else:
            self.parallel_dims, self.local_rank, self.global_rank = init_parallel_dims(
                titan_config
            )

        self.device = torch.device(f"cuda:{self.local_rank}")

        # Keep parameter storage in fp32 by default. This ensures AdamW can initialize
        # fp32 optimizer state after sharding if FSDP preserves fp32 parameter storage
        # under mixed precision.
        model = model.to(dtype=self.storage_dtype)

        # Optional random re-initialization to override all parameters
        reinit_model(model, trainer_args)

        parallel_strategy = titan_config.get("parallel_strategy", "qwen3")
        parallelize_fn = parallel_strategies[parallel_strategy]
        if isinstance(model, torch.nn.ModuleList):
            # Multi-stage pipeline parallelism
            for idx, m in enumerate(model):
                parallelize_fn(m, self.parallel_dims, titan_config, vpp_index=idx)
        else:
            model = parallelize_fn(model, self.parallel_dims, titan_config)
        self.model = model

        logger.info(model_summary(model))

        # Build optimizer and scheduler (after parallelization)
        self._build_optimizer_scheduler()

        # Load checkpoint if exists
        self._load_checkpoint(resume_path)

        # Disable automatic GC to prevent distributed straggler issues.
        # Random GC pauses on one rank stall all ranks at collectives.
        # We run a lightweight gen-1 collection every gc_freq steps instead.
        self.gc_freq = titan_config.get("gc_freq", 1000)
        gc.disable()
        gc.collect()
        logger.info(f"Disabled automatic GC, will collect every {self.gc_freq} steps")

        # Use TorchTitan's real loss mesh for token-count/stat reductions.
        # The 1-D "batch" mesh is created with a fake backend, so its process
        # group is not suitable for collectives such as all_reduce().
        loss_mesh = self.parallel_dims.get_optional_mesh("loss")
        self.dp_pg = loss_mesh.get_group() if loss_mesh is not None else None
        self.dp_size = loss_mesh.size() if loss_mesh is not None else 1

        # Log configuration
        wandb.config.update({"titan_config": titan_config})
        logger.info("Successfully initialized TitanTrainer with configuration:")
        logger.info(f"  FSDP enabled: {self.parallel_dims.fsdp_enabled}")
        logger.info(f"  dp_shard: {self.parallel_dims.dp_shard}")
        logger.info(f"  dp_replicate: {self.parallel_dims.dp_replicate}")
        logger.info(
            f"  gradient_accumulation_steps: {self.gradient_accumulation_steps}"
        )

    def _build_optimizer_scheduler(self):
        """Create optimizer and LR scheduler after parallelization.

        The only LR schedule supported is linear warmup → cosine decay →
        constant floor. This is the mainstream default in modern LLM
        training (GPT-3, LLaMA, Qwen, etc.) and is sufficient for virtually
        all SpeechLM training recipes, so we hard-code it here to keep the
        config surface small. Configurable knobs on ``trainer.lr_scheduler``:

        - ``warmup_steps`` (default 1000): linear ramp 0 → peak_lr.
        - ``decay_end_step`` (default ``trainer.max_step``): step at which
          cosine decay finishes and the LR holds at the floor.
        - ``min_lr_ratio`` (default 0.0): floor as a fraction of peak_lr.

        If a future recipe needs a different schedule (e.g. inverse-sqrt
        for encoder-decoder ASR), subclass this method.
        """
        opt_config = self.trainer_args.get("optimizer", {})
        lr_config = self.trainer_args.get("lr_scheduler", {})

        # Weight decay on all trainable parameters except input embeddings.
        # Embeddings are excluded because rare tokens that aren't seen often
        # would decay toward zero without gradient signal to restore them.
        decay_params = []
        no_decay_params = []
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            if "embed_tokens" in name:
                logger.info(f"No weight decay for: {name}")
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        total_params = sum(p.numel() for p in decay_params) + sum(
            p.numel() for p in no_decay_params
        )
        logger.info(f"Total trainable parameters: {total_params:,}")

        weight_decay = opt_config.get("weight_decay", 0.1)
        param_groups = [
            {"params": decay_params, "weight_decay": weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        # Build optimizer
        optimizer_name = opt_config.get("name", "AdamW")
        optimizer_cls = getattr(torch.optim, optimizer_name)

        optimizer_kwargs = {
            "lr": opt_config.get("lr", 1e-4),
            "betas": (opt_config.get("beta1", 0.9), opt_config.get("beta2", 0.95)),
            "eps": opt_config.get("eps", 1e-8),
        }

        # Use fused optimizer if available (faster on CUDA)
        if optimizer_name == "AdamW" and torch.cuda.is_available():
            optimizer_kwargs["fused"] = True

        self.optimizer = optimizer_cls(param_groups, **optimizer_kwargs)
        logger.info(
            f"Created {optimizer_name} optimizer with lr={optimizer_kwargs['lr']}, "
            f"weight_decay={weight_decay} (decay group only)"
        )

        # Build LR scheduler: linear warmup → cosine decay → constant
        # Example: warmup 2k steps to peak_lr, cosine decay to min_lr_ratio
        # at decay_end_step, then hold constant at min_lr_ratio * peak_lr
        warmup_steps = lr_config.get("warmup_steps", 1000)
        min_lr_ratio = lr_config.get("min_lr_ratio", 0.0)
        decay_end_step = lr_config.get("decay_end_step", self.max_step)

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step) / float(max(1, warmup_steps))
            elif step < decay_end_step:
                progress = (step - warmup_steps) / max(1, decay_end_step - warmup_steps)
                return min_lr_ratio + 0.5 * (1.0 - min_lr_ratio) * (
                    1.0 + math.cos(math.pi * progress)
                )
            else:
                return min_lr_ratio

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        logger.info(
            f"Created LR scheduler with warmup_steps={warmup_steps}, "
            f"min_lr_ratio={min_lr_ratio}, decay_end_step={decay_end_step}"
        )

    def _save_checkpoint(self, step: int) -> None:
        """Save a single DCP checkpoint containing model, optimizer, and metadata.

        Each rank writes only its own FSDP shard — no full-state gathering,
        so this scales to arbitrarily large models without OOM.

        Args:
            step: Current training step
        """
        checkpoint_dir = self.output_dir / "checkpoints" / f"step_{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "model": get_model_state_dict(self.model),
            "optimizer": get_optimizer_state_dict(
                self.model,
                self.optimizer,
                options=StateDictOptions(full_state_dict=False),
            ),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "global_step": self.global_step,
        }
        dcp.save(state, checkpoint_id=str(checkpoint_dir))
        dist.barrier()

        if self.global_rank == 0:
            logger.info(f"Saved checkpoint to {checkpoint_dir}")

    def _load_checkpoint(self, resume_path: Optional[Path]) -> None:
        """Load checkpoint from DCP, resuming model, optimizer, and LR state.

        Args:
            resume_path: Optional path to checkpoint directory
        """
        checkpoint_dir = None
        is_resume = False

        if resume_path and resume_path.exists():
            checkpoint_dir = resume_path
        elif (self.output_dir / "checkpoints").exists():
            ckpt_base = self.output_dir / "checkpoints"
            checkpoints = [
                d
                for d in ckpt_base.iterdir()
                if d.is_dir() and d.name.startswith("step_")
            ]
            if checkpoints:
                checkpoint_dir = sorted(
                    checkpoints,
                    key=lambda x: int(x.name.split("step_")[-1]),
                    reverse=True,
                )[0]
                is_resume = True

        if checkpoint_dir and checkpoint_dir.is_dir():
            logger.info(
                f"Loading checkpoint from {checkpoint_dir} "
                f"(mode={'resume' if is_resume else 'load_pretrained'})"
            )

            state = {
                "model": get_model_state_dict(self.model),
            }
            if is_resume:
                state["optimizer"] = get_optimizer_state_dict(
                    self.model,
                    self.optimizer,
                    options=StateDictOptions(full_state_dict=False),
                )
                state["lr_scheduler"] = self.lr_scheduler.state_dict()
                state["global_step"] = 0

            dcp.load(
                state,
                checkpoint_id=str(checkpoint_dir),
                planner=DefaultLoadPlanner(allow_partial_load=not is_resume),
            )

            set_model_state_dict(
                self.model,
                state["model"],
                options=StateDictOptions(strict=is_resume),
            )
            if is_resume:
                set_optimizer_state_dict(
                    self.model,
                    self.optimizer,
                    state["optimizer"],
                    options=StateDictOptions(
                        full_state_dict=False,
                        strict=is_resume,
                    ),
                )
                self.lr_scheduler.load_state_dict(copy.deepcopy(state["lr_scheduler"]))
                self.global_step = state["global_step"]

            logger.info(
                f"Loaded checkpoint: {checkpoint_dir} | step={self.global_step} | "
                f"lr_scheduler.last_epoch={self.lr_scheduler.last_epoch} | "
                f"actual_lr={self.optimizer.param_groups[0]['lr']}"
            )
        else:
            logger.info("No checkpoint found, starting from step 0")

    def _all_reduce_stats(
        self,
        stats: Dict[str, torch.Tensor],
        grad_accum: int = 1,
    ) -> None:
        """All-reduce statistics over the DP mesh with token-weighted averaging.

        Keys in ``count_normalized_keys`` are accumulated as raw sums
        (weighted by local token count) over micro-batches and then
        normalized by the global token count. Other keys are simple
        averages over DP ranks and micro-batches.

        Args:
            stats: Dictionary of statistics tensors.  Modified in-place.
                   Must contain ``"count"`` for token-weighted normalization.
            grad_accum: Number of micro-batches accumulated (used to average
                        non-loss stats).
        """
        if not dist.is_initialized():
            return

        handles = []
        for key in sorted(stats):
            if not isinstance(stats[key], torch.Tensor):
                stats[key] = torch.tensor(stats[key], device=self.device)
            handle = dist.all_reduce(
                stats[key],
                op=dist.ReduceOp.SUM,
                group=self.dp_pg,
                async_op=True,
            )
            handles.append((key, handle))

        for key, handle in handles:
            if handle is not None:
                handle.wait()

        count = stats.pop("count")
        for key in stats:
            if key in self.count_normalized_keys:
                stats[key] = stats[key] / count
            else:
                stats[key] = stats[key] / (self.dp_size * grad_accum)

    def run(self) -> None:
        """Main training loop."""
        logger.info(
            f"Starting training from step {self.global_step} to {self.max_step}"
        )

        while self.global_step < self.max_step:
            self.train()
            self.valid()

            # Save checkpoint
            self._save_checkpoint(self.global_step)

        logger.info("Training completed!")

    def train(self) -> None:
        """Execute one training epoch (save_interval optimizer steps).

        With gradient accumulation, each optimizer step consumes
        ``gradient_accumulation_steps`` micro-batches. The iterator is
        sized so that ``save_interval`` optimizer steps are performed.
        """
        self.model.train()
        grad_accum = self.gradient_accumulation_steps

        # Request enough micro-batches for save_interval optimizer steps.
        # Use global_step directly as batch offset (not multiplied by
        # grad_accum) so checkpoint resume is independent of grad_accum.
        iterator = self.train_data_factory.build_iter(
            global_step=self.global_step,
            length=self.save_interval * grad_accum,
        )
        data_iter = iter(iterator)

        for _ in range(self.save_interval):

            iter_start = time.time()
            self.optimizer.zero_grad(set_to_none=True)

            self.model.reset_loss_stats()
            stats = dict()
            for _micro in range(grad_accum):
                is_last_micro = _micro == grad_accum - 1

                # For HSDP with gradient accumulation: defer the all-reduce
                # to the last micro-batch so that intermediate reduce-scatter
                # outputs accumulate in float32 (partial_reduce_output) instead
                # of being cast to bfloat16 and accumulated in sharded param's
                # grad.  This reduces rounding error accumulation.
                self.model.set_requires_all_reduce(is_last_micro)
                self.model.set_is_last_backward(is_last_micro)

                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                batch = to_device(
                    batch, self.device, dtype=self.dtype, non_blocking=True
                )

                local_count = (batch["loss_masks"][:, :, 0] != 0).float().sum()
                global_count = local_count.clone()
                dist.all_reduce(global_count, op=dist.ReduceOp.SUM, group=self.dp_pg)
                batch["loss_scale"] = self.dp_size / (global_count * grad_accum)

                loss = self.model(**batch)
                loss.backward()

                # Accumulate raw-sum stats across micro-batches
                for key, val in batch["data_stats"].items():
                    stats[key] = stats.get(key, 0.0) + val

            # Gradient clipping (torchtitan version handles DTensor/FSDP2)
            grad_norm = dist_utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_norm,
                foreach=True,
                ep_enabled=self.parallel_dims.ep_enabled,
            )

            # Optimizer step
            self.optimizer.step()
            self.lr_scheduler.step()

            # Sync and log statistics
            stats.update(self.model._loss_stats)
            self._all_reduce_stats(stats, grad_accum=grad_accum)
            stats = {f"train/{k}": float(v.cpu()) for k, v in stats.items()}
            stats["train/lr"] = self.lr_scheduler.get_last_lr()[0]
            stats["train/grad_norm"] = (
                grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
            stats["time/iter"] = time.time() - iter_start

            # Log to wandb
            wandb.log(stats, step=self.global_step)

            # Console logging (rank 0 only, 4 significant digits).
            # NOTE(Jinchuan): This is for current step, not for average values
            # across steps.
            # To check the comprehensive performance, please use wandb.
            if self.global_rank == 0 and self.global_step % self.log_interval == 0:
                short = {k: f"{v:.4g}" for k, v in stats.items()}
                logger.info(f"step {self.global_step}, stats: {short}")

            # Periodic lightweight GC to reclaim memory without straggler stalls
            if self.global_step > 1 and self.global_step % self.gc_freq == 0:
                gc.collect(1)

            self.global_step += 1

    def valid(self) -> None:
        """Run validation on all validation datasets."""
        self.model.eval()

        for name, factory in self.valid_data_factories.items():
            iterator = factory.build_iter()

            local_len = torch.tensor(
                [len(iterator)],
                device=self.device,
                dtype=torch.long,
            )
            dist.all_reduce(local_len, op=dist.ReduceOp.MIN)
            num_valid_steps = int(local_len.item())

            stats = dict()
            self.model.reset_loss_stats()

            with torch.no_grad():
                for _, batch in zip(range(num_valid_steps), iterator):
                    batch = to_device(
                        batch, self.device, dtype=self.dtype, non_blocking=True
                    )
                    self.model(**batch)

                    for key, val in batch["data_stats"].items():
                        stats[key] = stats.get(key, 0.0) + val

            stats.update(self.model._loss_stats)
            if stats:
                self._all_reduce_stats(stats, grad_accum=num_valid_steps)
            all_stats = {
                f"val/{name}/{key}": float(value.cpu()) for key, value in stats.items()
            }
            wandb.log(all_stats, step=self.global_step)

            if self.global_rank == 0:
                short = {k: f"{v:.4g}" for k, v in all_stats.items()}
                logger.info(f"Validation [{name}]: {short}")
