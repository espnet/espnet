"""Definition of the low-rank adaptation (LoRA) for large models.

References:
    1. LoRA: Low-Rank Adaptation of Large Language Models
       (https://arxiv.org/pdf/2106.09685.pdf)
    2. https://github.com/microsoft/LoRA.git
    3. https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora.py

"""
from typing import List

import torch
from typeguard import check_argument_types
from espnet2.layers.houlsby_adapter_layer import HoulsbyTransformerSentenceEncoderLayer
from espnet2.asr.frontend.s3prl import S3prlFrontend
try:
    import loralib as lora
except Exception:
    lora = None
    
try:
    import s3prl 
except Exception:
    s3prl = None
    
def create_adapter(
    model: torch.nn.Module,
    adapter: str,
    adapter_conf: dict,
):
    """Create adapter for the base model.


    Args:
        model (torch.nn.Module): Base model to be adapted.
        adapter_type (str): Name of adapter
        adapter_conf (dict): Configuration for the adapter
            e.g.  {"rank": 8, "alpha": 8, ...} for lora

    """
    assert check_argument_types()
    
    if adapter == 'lora':
        create_adapter_fn = create_lora_adapter
    
    elif adapter == 'houlsby':
        create_adapter_fn = create_houlsby_adapter
    
    else:
        raise NotImplementedError(f"Adapter {adapter} is not supported.")
    create_adapter_fn(model=model, **adapter_conf)

    
def create_houlsby_adapter(
    model: torch.nn.Module,
    bottleneck: int = 32,
    target_layers: List[int] = [],
):
    assert check_argument_types()
    assert hasattr(model, 'frontend') and isinstance(model.frontend, S3prlFrontend), "Only support S3PRL frontend now !!"
    if s3prl is None:
        print("Error: S3PRL is not properly installed.")
        print("Please install S3PRL: cd ${MAIN_ROOT}/tools && make s3prl.done")
        raise RuntimeError("Requiring S3PRL. ")
    
    is_traget_layer_exists = False
    key_list = [key for key, _ in model.named_modules()]
    num_layers = model.frontend.upstream.num_layers -1
    if len(target_layers) == 0:
        target_layers = list(range(num_layers))
        
    for layer_idx in target_layers:

        key = f"frontend.upstream.upstream.model.encoder.layers.{layer_idx}"
        if key not in key_list:
            continue
        
        is_traget_layer_exists = True
        parent_module, target_name, target_module = get_submodules(model, key)
        new_module = create_new_houlsby_module(target_module, bottleneck)
        new_module.to(next(target_module.parameters()).device)
        setattr(parent_module, target_name, new_module)
    
    if not is_traget_layer_exists:
        raise ValueError(
            f"Target layers {target_layers} not found in the base model."
        )

def create_lora_adapter(
    model: torch.nn.Module,
    rank: int = 8,
    alpha: int = 8,
    dropout_rate: float = 0.0,
    target_modules: List[str] = ["query"],
    bias_type: str = "none",
):
    """Create LoRA adapter for the base model.

    See: https://arxiv.org/pdf/2106.09685.pdf

    Args:
        model (torch.nn.Module): Base model to be adapted.
        rank (int): Rank of LoRA matrices. Defaults to 8.
        alpha (int): Constant number for LoRA scaling. Defaults to 8.
        dropout_rate (float): Dropout probability for LoRA layers. Defaults to 0.0.
        target_modules (List[str]): List of module(s) to apply LoRA adaptation.
            e.g. ["query", "key", "value"] for all layers,
            while ["encoder.encoders.blocks.0.attn.key"] for a specific layer.
        bias_type (str): Bias training type for LoRA adaptaion, can be
            one of ["none", "all", "lora_only"].
            "none" means not training any bias vectors;
            "all" means training all bias vectors, include LayerNorm biases;
            "lora_only" means only training bias vectors in LoRA adapted modules.

   
    """
        
    assert check_argument_types()

    if lora is None:
        raise RuntimeError(
            "Requiring loralib. Install loralib following: "
            "https://github.com/microsoft/LoRA"
        )

    is_traget_module_exists = False
    key_list = [key for key, _ in model.named_modules()]

    for key in key_list:
        # TODO endswith may not be a good choice
        # exists maybe better in our case cuz our class won't end in the key
        # 
        if not check_target_module_exists(key, target_modules):
            continue

        is_traget_module_exists = True

        parent_module, target_name, target_module = get_submodules(model, key)
        # TODO Replace lora.LoRALayer with assigned instance
        # Eg. AdapterEncoderLayer for adapter case
        if not isinstance(target_module, lora.LoRALayer):
            new_module = create_new_lora_module(target_module, rank, alpha, dropout_rate)
            replace_module(parent_module, target_name, target_module, new_module)
        else:
            continue

    if not is_traget_module_exists:
        raise ValueError(
            f"Target modules {target_modules} not found in the base model."
        )

    lora.mark_only_lora_as_trainable(model, bias_type)


def check_target_module_exists(key: str, target_modules: List[str]):
    """Check if the target_modules matchs the given key."""

    return any([key.endswith(target_key) for target_key in target_modules])


def get_submodules(model: torch.nn.Module, key: str):
    """Return the submodules of the given key."""

    parent_module = model.get_submodule(".".join(key.split(".")[:-1]))
    target_name = key.split(".")[-1]
    target_module = model.get_submodule(key)
    return parent_module, target_name, target_module

def create_new_houlsby_module(
    target_module: torch.nn.Module, bottleneck: int
):
    """Create a new houlsby adapter module for the given target TransformerSentenceEncoderLayer module."""
    embedding_dim = target_module.embedding_dim
    ffn_embedding_dim = target_module.fc1.out_features
    num_attention_heads = target_module.self_attn.num_heads
    dropout = target_module.dropout1.p
    attention_dropout = target_module.self_attn.dropout_module.p
    activation_dropout = target_module.dropout2.p
    activation_fn = target_module.activation_fn.__name__
    layer_norm_first = target_module.layer_norm_first

    # initialize adapter-added transformer layer
    adapter_added_layer = HoulsbyTransformerSentenceEncoderLayer(
        embedding_dim=embedding_dim,
        ffn_embedding_dim=ffn_embedding_dim,
        num_attention_heads=num_attention_heads,
        dropout=dropout,
        attention_dropout=attention_dropout,
        activation_dropout=activation_dropout,
        activation_fn=activation_fn,
        layer_norm_first=layer_norm_first,
        bottleneck=bottleneck,
    )
    
    # Get default requires_grad 
    for n, p in adapter_added_layer.named_parameters():
        if 'adapter' in n:
            continue
        p.requires_grad = eval(f"target_module.{n}").requires_grad
        
    # copy weights from the target module
    orig_state_dict = target_module.state_dict()
    adapter_added_layer.load_state_dict(orig_state_dict, strict=False)
    
    # Copy all hooks to the new layer
    for k, v in target_module.__dict__.items():
        if 'hook' not in k:
            continue
        adapter_added_layer.__dict__[k] = v
        
    return adapter_added_layer
def create_new_lora_module(
    target_module: torch.nn.Module, rank: int, alpha: int, dropout_rate: float
):
    """Create a new lora module for the given target module."""

    bias = hasattr(target_module, "bias") and target_module.bias is not None

    if isinstance(target_module, torch.nn.Embedding):
        new_module = lora.Embedding(
            target_module.num_embeddings,
            target_module.embedding_dim,
            r=rank,
            lora_alpha=alpha,
        )
    elif isinstance(target_module, torch.nn.Linear):
        new_module = lora.Linear(
            target_module.in_features,
            target_module.out_features,
            bias=bias,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout_rate,
        )
    else:
        raise ValueError(
            f"Target module {target_module} is not supported. "
            f"Currently, only `torch.nn.Embedding`, `torch.nn.Conv2d` "
            f"`torch.nn.Linear` and are supported."
        )

    return new_module


def replace_module(
    parent_module: torch.nn.Module,
    child_name: str,
    old_module: torch.nn.Module,
    new_module: torch.nn.Module,
):
    """Replace the target module with the new module."""
    # TODO add hook and whether requires_grad to them
    device = old_module.weight.device
    setattr(parent_module, child_name, new_module)

    # copy weight and bias from the target module
    new_module.weight = old_module.weight
    if hasattr(old_module, "bias") and old_module.bias is not None:
        new_module.bias = old_module.bias

    # move the new_module to the same device as the old_module
    new_module.to(device)
