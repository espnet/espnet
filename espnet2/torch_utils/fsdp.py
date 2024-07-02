#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# NOTE (Jinchuan): Pytorch built-in FullyShardedDataParallel, a beta feature
# FSDP will have higher training throughput given a large number of GPUs
# and sufficient communication bandwidth. However, it will have extra
# requirements for model architecture.
# Before using this feature, make sure you read the following toturials about FSDP:
# https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
# https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html
# https://pytorch.org/docs/stable/fsdp.html

# NOTE (Jinchuan): Pytorch APIs of FSDP is subjected to rapid change. We intend to
# supprot this FSDP feature from torch 2.0.1+ .
# Current code can work on the following torch versions:
#   - 2.0.1
#   - 2.3.0
# Please raise an issue in https://github.com/espnet/espnet if this code doesn't work
# on specific torch version above 2.0.1

import torch
import functools
from packaging.version import parse as V
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
)

try:
    from torch.distributed.fsdp import FullOptimStateDictConfig
except:  # torch 2.0.1
    from torch.distributed.fsdp.api import FullOptimStateDictConfig


def sum_parameter(module: torch.nn.Module):
    return sum([p.numel() for p in module.parameters()])


def transformer_auto_wrap_policy(
    module_cls_list,
    module: torch.nn.Module,
    recurse: bool = False,
    nonwrapped_numel: int = -1,
    min_num_params: int = 30 * 1e6,
):
    if recurse:  # always be recursive to the children
        return True
    else:
        for module_cls in module_cls_list:
            if (
                isinstance(module, module_cls)
                and sum_parameter(module) > min_num_params
            ):
                return True
        return False


def warp_fsdp(
    model: torch.nn.Module, use_amp: bool = False, min_num_params: int = 30 * 1e6
):
    # auto_warp_policy
    # NOTE (Jinchuan): the model should have this layer_cls attribute to indicate
    # which modules should be warpped by FSDP. We strongly recommend users to only
    # specify this as the model layer definition (typically Transformer layer).
    # Fail to do so may leads to unexpected behavior. See FSDP official documents
    # for more details.
    if not hasattr(model, "layer_cls") or len(model.layer_cls) == 0:
        raise ValueError("Specify the layer_cls feature in model to use FSDP")
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        module_cls_list=model.layer_cls,
        min_num_params=min_num_params,
    )

    # precison
    # NOTE (Jinchuan): this is
    if V(torch.__version__) >= V("1.10.0") and torch.cuda.is_available() and use_amp:
        if torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
    else:
        dtype = torch.float32
    mixed_precision = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )

    # NOTE(Jinchuan) Since our models are usually not very large, we currently
    # don't consider more advanced choices such as cpu_offload etc.
    # sync_module_states=True: in case a pre-trained model is loaded.
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sync_module_states=True,
        use_orig_params=True
    )


def get_model_and_optimizer_state_dict_fsdp(model, optimizers):
    """get model and optimizer state dict when the model is warpped by FSDP"""
    FSDP.set_state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    )
    state_dict = model.state_dict()

    if len(optimizers) > 1:
        raise ValueError(f"currently FSDP can only support one optimizer")
    # NOTE(Jinchuan): will cause segment fault when not all parameters
    # require gradients. In this case, just skip saving the optimizer
    # states. This should be roughly ok as this is usually for small-scale
    # fine-tuning.
    if all(p.requires_grad for p in model.parameters()):
        optim_state_dict = [FSDP.optim_state_dict(model, optimizers[0])]
    else:
        optim_state_dict = {}

    return state_dict, optim_state_dict


def prepare_for_resume_fsdp(states, model, optimizers):
    """modify the optimizer states so it can be loaded into a optimizer
    over the sharded parameters.
    """
    if len(optimizers) > 1:
        raise ValueError(f"currently FSDP can only support one optimizer")

    states["optimizers"][0] = FSDP.optim_state_dict_to_load(
        states["optimizers"][0],
        model,
        optimizers[0],
    )

    return states
