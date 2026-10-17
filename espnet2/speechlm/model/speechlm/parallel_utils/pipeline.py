# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Pipeline parallel utilities for SpeechLM models.

Wraps PP stage model(s) as PipelineStage(s) and builds a PP schedule with
support for pre-split microbatch feeding.

Supports both single-stage schedules (1F1B — one stage per rank) and
multi-stage schedules (Interleaved1F1B — multiple virtual stages per rank).
"""

import logging
from typing import Any, Dict, Tuple, Union

import torch.nn as nn
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    get_schedule_class,
)
from torchtitan.distributed import ParallelDims

logger = logging.getLogger(__name__)


def build_pipeline(
    model: Union[nn.Module, nn.ModuleList],
    parallel_dims: ParallelDims,
    titan_config: Dict[str, Any],
    n_microbatches: int,
) -> Tuple[Any, bool]:
    """Build PipelineStage(s) and PP schedule for stage model(s).

    For single-stage schedules (e.g. 1F1B), ``model`` is a single
    nn.Module. For multi-stage schedules (e.g. Interleaved1F1B),
    ``model`` is an nn.ModuleList of model chunks (one per virtual
    stage on this rank).

    Args:
        model: Single PP stage model, or nn.ModuleList of model chunks
            for interleaved schedules. Each model must have ``stage_idx``,
            ``num_virtual_stages``, ``pp_rank``, ``pp_degree``, and
            ``is_last_stage`` attributes.
        parallel_dims: TorchTitan ParallelDims with PP mesh built.
        titan_config: Config dict. Reads ``pp_schedule`` (default "1F1B").
        n_microbatches: Number of microbatches per optimizer step.

    Returns:
        ``(schedule, pp_has_last_stage)`` tuple.
    """
    schedule_name = titan_config.get("pp_schedule", "1F1B")
    schedule_cls = get_schedule_class(schedule_name)
    is_multi = issubclass(schedule_cls, PipelineScheduleMulti)

    pp_mesh = parallel_dims.get_mesh("pp")

    def _identity_loss(output, target):
        """The last stage forward already returns the scalar loss.

        The schedule normalizes the stage output to a tuple, so we
        unwrap to get the scalar tensor for backward and losses list.
        """
        if isinstance(output, tuple):
            return output[0]
        return output

    if is_multi:
        assert isinstance(model, nn.ModuleList) and len(model) > 1, (
            f"Multi-stage schedule {schedule_name} requires an "
            f"nn.ModuleList with >1 model chunks, got {type(model)} "
            f"with {len(model) if isinstance(model, nn.ModuleList) else 1} chunks"
        )
        vpp_degree = len(model)
        assert n_microbatches % vpp_degree == 0, (
            f"n_microbatches ({n_microbatches}) must be divisible by "
            f"vpp_degree ({vpp_degree}) for schedule {schedule_name}"
        )

        stages = []
        for chunk in model:
            device = next(chunk.parameters()).device
            stage = PipelineStage(
                chunk,
                stage_index=chunk.stage_idx,
                num_stages=chunk.num_virtual_stages,
                device=device,
                group=pp_mesh.get_group(),
            )
            stages.append(stage)

        schedule = schedule_cls(
            stages,
            n_microbatches=n_microbatches,
            loss_fn=_identity_loss,
            scale_grads=False,
        )
        pp_has_last_stage = any(chunk.is_last_stage for chunk in model)

        logger.info(
            f"Built PP pipeline (multi-stage): schedule={schedule_name}, "
            f"n_microbatches={n_microbatches}, "
            f"stages_on_rank={[c.stage_idx for c in model]}, "
            f"last_stage={pp_has_last_stage}"
        )
    else:
        if isinstance(model, nn.ModuleList):
            assert len(model) == 1, (
                f"Single-stage schedule {schedule_name} expects 1 model "
                f"chunk, got {len(model)}"
            )
            model = model[0]

        device = next(model.parameters()).device
        stage = PipelineStage(
            model,
            stage_index=model.pp_rank,
            num_stages=model.pp_degree,
            device=device,
            group=pp_mesh.get_group(),
        )

        schedule = schedule_cls(
            stage,
            n_microbatches=n_microbatches,
            loss_fn=_identity_loss,
            scale_grads=False,
        )
        pp_has_last_stage = model.is_last_stage

        logger.info(
            f"Built PP pipeline: schedule={schedule_name}, "
            f"n_microbatches={n_microbatches}, "
            f"stage={model.pp_rank}/{model.pp_degree}, "
            f"last_stage={pp_has_last_stage}"
        )

    # Override _split_inputs to pass through pre-split microbatch lists.
    # The trainer passes (arg_mbs, kwarg_mbs) as the two positional args
    # to step(). Inside step(), these become args=(arg_mbs, kwarg_mbs).
    # _split_inputs must return (args_split, kwargs_split) where
    # args_split is the list of per-microbatch arg tuples and
    # kwargs_split is the list of per-microbatch kwarg dicts.
    # Works identically for both PipelineScheduleSingle and
    # PipelineScheduleMulti since both call _split_inputs the same way.
    schedule._split_inputs = lambda args, kwargs: (args[0], args[1])

    return schedule, pp_has_last_stage
