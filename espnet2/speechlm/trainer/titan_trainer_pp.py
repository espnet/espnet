# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Pipeline-parallel trainer extending TitanTrainer.

Overrides train(), valid(), and checkpoint save/load with PP-aware
implementations. self.model is always an nn.ModuleList of model chunks
(even for single-stage 1F1B, it is a 1-item list). This keeps all
iteration logic uniform.

All other functionality (optimizer, scheduler, logging, GC) is
inherited from TitanTrainer.
"""

import copy
import gc
import logging
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
import torch.nn as nn
import wandb
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torchtitan.distributed import utils as dist_utils

from espnet2.speechlm.model.speechlm.parallel_utils import build_pipeline
from espnet2.speechlm.trainer.titan_trainer import TitanTrainer
from espnet2.speechlm.utils.data import to_device

logger = logging.getLogger(__name__)


class TitanPPTrainer(TitanTrainer):
    """Pipeline-parallel trainer.

    Inherits from TitanTrainer and overrides train() and valid() to use
    the PP schedule for forward/backward instead of explicit gradient
    accumulation.

    ``self.model`` is always an ``nn.ModuleList`` of model chunks. For
    single-stage schedules (1F1B) it contains one element; for
    multi-stage schedules (Interleaved1F1B) it contains ``vpp_degree``
    elements. The last element is always the last virtual stage (which
    computes the loss).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        if not isinstance(self.model, nn.ModuleList):
            self.model = nn.ModuleList([self.model])

        titan_config = self.trainer_args.get("titan_config", {})
        self.pp_schedule, self.pp_has_last_stage = build_pipeline(
            self.model,
            self.parallel_dims,
            titan_config,
            n_microbatches=self.gradient_accumulation_steps,
        )

        pp_mesh = self.parallel_dims.get_mesh("pp")
        self.pp_pg = pp_mesh.get_group()
        self.pp_degree = pp_mesh.size()
        self.pp_rank = pp_mesh.get_local_rank()

    def train(self) -> None:
        """PP training: schedule.step() handles microbatching and backward."""
        self.model.train()
        n_microbatches = self.gradient_accumulation_steps

        iterator = self.train_data_factory.build_iter(
            global_step=self.global_step,
            length=self.save_interval * n_microbatches,
        )
        data_iter = iter(iterator)

        for _ in range(self.save_interval):
            iter_start = time.time()
            self.optimizer.zero_grad(set_to_none=True)

            stats = dict()
            for m in self.model:
                m.reset_loss_stats()

            kwarg_mbs = []
            for _ in range(n_microbatches):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    break
                kwarg_mbs.append(
                    to_device(batch, self.device, dtype=self.dtype, non_blocking=True)
                )

            for mb in kwarg_mbs:
                local_count = (mb["loss_masks"][:, :, 0] != 0).float().sum()
                global_count = local_count.clone()
                dist.all_reduce(global_count, op=dist.ReduceOp.SUM, group=self.dp_pg)
                mb["loss_scale"] = self.dp_size / (global_count * len(kwarg_mbs))

                for key, val in mb.get("data_stats", {}).items():
                    stats[key] = stats.get(key, 0.0) + val

            # NOTE(Jinchuan): Follow the input format, many dummy arguments are needed.
            arg_mbs = [()] * len(kwarg_mbs)
            dummy_target = torch.zeros(
                len(kwarg_mbs),
                device=self.device,
                dtype=self.dtype,
            )
            losses = []
            self.pp_schedule.step(
                arg_mbs,
                kwarg_mbs,
                target=dummy_target,
                losses=losses,
                return_outputs=False,
            )

            grad_norm = dist_utils.clip_grad_norm_(
                self.model.parameters(),
                self.max_norm,
                foreach=True,
                pp_mesh=self.parallel_dims.get_optional_mesh("pp"),
                ep_enabled=self.parallel_dims.ep_enabled,
            )

            self.optimizer.step()
            self.lr_scheduler.step()

            # (1) Last-stage ranks: collect loss stats and all-reduce over DP peers
            if self.pp_has_last_stage:
                stats.update(self.model[-1]._loss_stats)
                self._all_reduce_stats(stats, grad_accum=n_microbatches)

            # (2) Broadcast normalized stats from last PP stage to all PP ranks
            # so that rank 0 (first stage) can log them to wandb.
            stats = self._broadcast_stats_across_pp(stats)

            stats = {f"train/{k}": v for k, v in stats.items()}
            stats["train/lr"] = self.lr_scheduler.get_last_lr()[0]
            stats["train/grad_norm"] = (
                grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
            )
            stats["time/iter"] = time.time() - iter_start

            wandb.log(stats, step=self.global_step)

            if self.global_rank == 0 and self.global_step % self.log_interval == 0:
                short = {k: f"{v:.4g}" for k, v in stats.items()}
                logger.info(f"step {self.global_step}, stats: {short}")

            if self.global_step > 1 and self.global_step % self.gc_freq == 0:
                gc.collect(1)

            self.global_step += 1

    def valid(self) -> None:
        pass  # TODO(Jinchuan): Implement validation loop

    def _broadcast_stats_across_pp(self, stats: Dict) -> Dict[str, float]:
        """Broadcast stats from the last PP stage to all PP stages.

        All ranks compute the same key set locally (count_normalized_keys
        is a class constant, data_stats keys are identical across ranks),
        then broadcast each value from the last PP rank.

        Returns:
            Dictionary with all values converted to Python floats.
        """
        pp_src = self.pp_degree - 1
        keys = sorted(set(self.count_normalized_keys) | set(stats.keys()))

        result = {}
        for key in keys:
            val = stats.get(key, 0.0)
            t = (
                torch.tensor(val, device=self.device)
                if not isinstance(val, torch.Tensor)
                else val.clone()
            )
            dist.broadcast(t, group_src=pp_src, group=self.pp_pg)
            result[key] = float(t.cpu())
        return result

    # ----------------------------------------------------------------
    # Checkpointing: per-PP-stage subdirectories
    # ----------------------------------------------------------------

    def _pp_checkpoint_dir(self, step: int) -> Path:
        """Checkpoint dir for this PP stage: step_{step}/pp_{rank}_{degree}."""
        base = self.output_dir / "checkpoints" / f"step_{step}"
        return base / f"pp_{self.pp_rank:02d}_{self.pp_degree:02d}"

    def _pp_process_group(self):
        """DP process group (same-stage peers) for DCP coordination.

        With HSDP (dp_replicate > 1), includes both dp_replicate and
        fsdp dimensions so all DP ranks for this PP stage participate.
        """
        if self.parallel_dims.dp_replicate_enabled:
            return self.parallel_dims.get_mesh(["dp_replicate", "fsdp"]).get_group()
        return self.parallel_dims.get_mesh("fsdp").get_group()

    def _save_checkpoint(self, step: int) -> None:
        """Save per-PP-stage DCP checkpoint."""
        checkpoint_dir = self._pp_checkpoint_dir(step)
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
            "pp_rank": self.pp_rank,
            "pp_degree": self.pp_degree,
        }
        dcp.save(
            state,
            checkpoint_id=str(checkpoint_dir),
            process_group=self._pp_process_group(),
        )
        dist.barrier()

        if self.global_rank == 0:
            logger.info(f"Saved PP checkpoint to {checkpoint_dir}")

    def _load_checkpoint(self, resume_path: Optional[Path]) -> None:
        """Load per-PP-stage DCP checkpoint."""
        checkpoint_dir = None

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

        if checkpoint_dir is None or not checkpoint_dir.is_dir():
            logger.info("No checkpoint found, starting from step 0")
            return

        pp_subdir = checkpoint_dir / (f"pp_{self.pp_rank:02d}_{self.pp_degree:02d}")
        if pp_subdir.is_dir():
            checkpoint_dir = pp_subdir
        else:
            logger.warning(
                f"PP subdirectory not found: {pp_subdir}. "
                f"Loading from {checkpoint_dir} directly."
            )

        logger.info(f"Loading PP checkpoint from {checkpoint_dir}")

        state = {
            "model": get_model_state_dict(self.model),
            "optimizer": get_optimizer_state_dict(
                self.model,
                self.optimizer,
                options=StateDictOptions(full_state_dict=False),
            ),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "global_step": 0,
        }
        dcp.load(
            state,
            checkpoint_id=str(checkpoint_dir),
            process_group=self._pp_process_group(),
        )

        set_model_state_dict(
            self.model,
            state["model"],
            options=StateDictOptions(strict=True),
        )
        set_optimizer_state_dict(
            self.model,
            self.optimizer,
            state["optimizer"],
            options=StateDictOptions(full_state_dict=False),
        )
        self.lr_scheduler.load_state_dict(copy.deepcopy(state["lr_scheduler"]))
        self.global_step = state["global_step"]

        logger.info(
            f"Loaded PP checkpoint: {checkpoint_dir} | "
            f"step={self.global_step} | pp_rank={self.pp_rank}"
        )
