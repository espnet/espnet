import torch
import functools
from packaging.version import parse as V

def warp_fsdp(model: torch.nn.Module, use_amp: bool = False):
    # Note(Jinchuan): Pytorch built-in FullyShardedDataParallel, a beta feature
    # FSDP will have higher training performance given a large number of GPUs
    # and sufficient communication bandwidth. However, it will have extra 
    # requirements for model architecture.
    # Before using this feature, make sure you read the following documents
    # https://pytorch.org/tutorials/intermediate/FSDP_tutorial.html
    # https://pytorch.org/tutorials/intermediate/FSDP_adavnced_tutorial.html
    # https://pytorch.org/docs/stable/fsdp.html
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
        from torch.distributed.fsdp import MixedPrecision
    except ImportError:
        raise ImportError("Your pytorch doesn't support FSDP, try to upgrade")
    
    # auto_warp_policy
    # Note(Jinchuan): we only apply FSDP to layers (typically transformer layers)
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
        transformer_layer_cls=set(model.layer_cls)
    )

    # precison
    # Note(Jinchuan): when amp is not applied, FSDP use float32; otherwise
    # bfloat16 is adopted. Please note Espnet supports amp training only with
    # bfloat16. The FSDP may possibly need a new scaler when bfloat16 is adopted.
    # According to:
    # https://github.com/pytorch/pytorch/issues/76607#issuecomment-1967021369
    # the original scaler can still be used.
    if (
        V(torch.__version__) >= V("1.10.0")
        and torch.cuda.is_available()
        and torch.cuda.is_bf16_supported()
        and use_amp
    ):
        dtype = torch.bfloat16
    else:
        dtype = torch.float32
    mixed_precision = MixedPrecision(
        param_dtype=dtype,
        reduce_dtype=dtype,
        buffer_dtype=dtype,
    )

    # Note(Jinchuan) Since our models are usually not very large, we currently 
    # don't consider more advanced choices such as cpu_offload.
    return FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,      
    )