#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

# NOTE (Jinchuan): Pytorch built-in FullyShardedDataParallel, a beta feature
# FSDP will have higher training performance given a large number of GPUs
# and sufficient communication bandwidth. However, it will have extra 
# requirements for model architecture.
# Before using this feature, make sure you read the following documents
# https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
# https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html
# https://pytorch.org/docs/stable/fsdp.html

# NOTE (Jinchuan): The code is based on Pytorch 2.0.1. Pytorch FSDP APIs
# are subjected to rapid change. We mainly follow this document:
# https://pytorch.org/docs/2.0/fsdp.html?highlight=fsdp#module-torch.distributed.fsdp

import torch
import functools
from packaging.version import parse as V
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    StateDictType,
    FullStateDictConfig,
)
from torch.distributed.fsdp.api import (
    FullOptimStateDictConfig,
    CPUOffload,
)

def sum_parameter(module: torch.nn.Module):
    total_sum = 0
    for param in module.parameters():
        total_sum += param.numel()
    return total_sum

def transformer_auto_wrap_policy(
    module_cls_list,
    module: torch.nn.Module,
    recurse: bool = False,
    nonwrapped_numel: int = -1,
    min_num_params: int = 30 * 1e6,
):
    if recurse: # always be recursive to the children
        return True
    else:
        for module_cls in module_cls_list:
            if isinstance(module, module_cls) and sum_parameter(module) > min_num_params:
                return True
        return False

def warp_fsdp(model: torch.nn.Module, use_amp: bool = False, min_num_params: int = 30 * 1e6):
    # auto_warp_policy
    # NOTE (Jinchuan): we only apply FSDP to layers (typically transformer layers)
    # but not for the other modules, such as embeddings. The remained modules
    # are usually small so applying FSDP to them may not have too much benefit. 
    # They would have some parameter-sharing strategy during training, which is not 
    # allowed in FSDP.
    if not hasattr(model, "layer_cls"):
        raise ValueError("Specify the layer_cls feature in model to use FSDP")
    if len(model.layer_cls) == 0:
        raise ValueError("layer_cls for model is empty. Cannot use FSDP")
    auto_wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        module_cls_list=model.layer_cls,
        min_num_params=min_num_params,
    )

    # precison
    # NOTE (Jinchuan): this is 
    if (
        V(torch.__version__) >= V("1.10.0")
        and torch.cuda.is_available()
        and use_amp
    ):
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
    # don't consider more advanced choices such as cpu_offload.
    # sync_module_states=True: in case a pre-trained model is loaded.
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        sync_module_states=True,
        cpu_offload=CPUOffload(offload_params=False),
    )

def get_model_and_optimizer_state_dict_fsdp(model, optimizers):
    """ get model and optimizer state dict when the model is warpped by FSDP """
    FSDP.set_state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        FullStateDictConfig(offload_to_cpu=True, rank0_only=True),
        FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=True),
    )
    state_dict = model.state_dict()

    if len(optimizers) > 1:
        raise ValueError(f"currently FSDP can only support one optimizer")
    optim_state_dict = [FSDP.optim_state_dict(model, optimizers[0])]

    return state_dict, optim_state_dict

def prepare_for_resume_fsdp(states, model, optimizers):
    """ modify the optimizer states so it can be loaded into a optimizer 
        over the sharded parameters.
    """
    if len(optimizers) > 1:
        raise ValueError(f"currently FSDP can only support one optimizer")
    optimizer = optimizers[0]

    states['optimizers'][0] = FSDP.optim_state_dict_to_load(
        states['optimizers'][0],
        model,
        optimizer,
    )

    return states