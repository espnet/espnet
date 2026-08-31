from typing import List, Optional

import torch
from typeguard import typechecked

from espnet2.layers.create_adapter_utils import (
    check_target_module_exists,
    get_submodules,
    replace_module,
)
from espnet2.layers.houlsby_adapter_layer import (
    Houlsby_Adapter,
    HoulsbyTransformerSentenceEncoderLayer,
)
from espnet2.layers.lora import LINEAR_BACKENDS
from espnet2.layers.lora import Embedding as LoraEmbedding
from espnet2.layers.lora import LoRALayer

try:
    from transformers.models.wav2vec2.modeling_wav2vec2 import (
        Wav2Vec2EncoderLayerStableLayerNorm,
    )

    is_transformers_available = True
except ImportError:
    is_transformers_available = False

try:
    import s3prl  # noqa
    from s3prl.upstream.wav2vec2.wav2vec2_model import TransformerSentenceEncoderLayer

    is_s3prl_available = True
except ImportError:
    is_s3prl_available = False


@typechecked
def create_houlsby_adapter(
    model: torch.nn.Module,
    bottleneck: int = 32,
    target_layers: List[int] = [],
):
    if not is_transformers_available:
        raise ImportError(
            "`transformers` is not available. Please install it via `pip install"
            " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
            " && ./installers/install_transformers.sh`."
        )
    if not is_s3prl_available:
        raise ImportError(
            "Error: S3PRL is not properly installed."
            "Please install S3PRL: cd ${MAIN_ROOT}/tools && make s3prl.done"
        )
    # Deferred import: ESPnet's S3prlFrontend pulls in a heavy dependency tree
    # that breaks `create_lora_adapter` if it fails to import. Only the
    # Houlsby path needs it.
    from espnet2.asr.frontend.s3prl import S3prlFrontend

    assert hasattr(model, "frontend") and isinstance(
        model.frontend, S3prlFrontend
    ), "Only support S3PRL frontend now !!"

    is_traget_layer_exists = False
    key_list = [key for key, _ in model.named_modules()]
    num_layers = model.frontend.upstream.num_layers - 1
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
        raise ValueError(f"Target layers {target_layers} not found in the base model.")


@typechecked
def create_lora_adapter(
    model: torch.nn.Module,
    rank: int = 8,
    alpha: int = 8,
    dropout_rate: float = 0.0,
    target_modules: List[str] = ["query"],
    bias_type: Optional[str] = "none",
    adapter_type: str = "lora",
    **kwargs,
):
    """Create a LoRA-family adapter for the base model.

    The backend used for ``torch.nn.Linear`` modules is selected by
    ``adapter_type``. Supported values (see :mod:`espnet2.layers.lora`):

        * ``"lora"``  -- vanilla LoRA (Hu et al., 2021).
        * ``"dora"``  -- DoRA (Liu et al., 2024).
        * ``"pissa"`` -- PiSSA (Meng et al., 2024).
        * ``"svft"``  -- SVFT (Lingam et al., 2024).
        * ``"ssvd"``  -- SSVD (Wang et al., ASRU 2025).

    ``nn.Embedding`` modules always receive the plain LoRA embedding
    adapter.

    Args:
        model: Base model to be adapted (mutated in place).
        rank: Rank of the LoRA matrices. Defaults to 8.
        alpha: Scaling constant for LoRA updates. Defaults to 8.
        dropout_rate: Dropout probability for LoRA layers.
        target_modules: List of module suffix strings to apply the adapter to,
            e.g. ``["linear_q"]`` or ``["encoder.encoders.0.attn.linear_q"]``.
        bias_type: Bias-training policy. One of ``"none"``, ``"all"``,
            ``"lora_only"``.
        adapter_type: Backend selector for adapted Linear modules.
        **kwargs: Backend-specific keyword arguments forwarded to the layer
            constructor (e.g. ``rotation_ratio`` for ``"ssvd"`` or
            ``off_diag`` for ``"svft"``).
    """

    is_traget_module_exists = False
    key_list = [key for key, _ in model.named_modules()]

    for key in key_list:
        if not check_target_module_exists(key, target_modules):
            continue

        is_traget_module_exists = True

        parent_module, target_name, target_module = get_submodules(model, key)
        if isinstance(target_module, LoRALayer):
            continue
        new_module = create_new_lora_module(
            target_module, rank, alpha, dropout_rate, adapter_type, **kwargs
        )
        replace_module(parent_module, target_name, target_module, new_module)

    if not is_traget_module_exists:
        raise ValueError(
            f"Target modules {target_modules} not found in the base model."
        )

    # Set the model (originally in train mode) to eval mode.
    # This step can avoid merging LoRA weights again
    # when loading pre-trained checkpoints.
    model.eval()


@typechecked
def create_new_houlsby_module(target_module: torch.nn.Module, bottleneck: int):
    """Create a new houlsby adapter module for the given target module.

    Currently, only support:
    Wav2Vec2EncoderLayerStableLayerNorm &
    TransformerSentenceEncoderLayer
    """
    if isinstance(target_module, Wav2Vec2EncoderLayerStableLayerNorm):

        input_size = target_module.layer_norm.normalized_shape[0]
        target_module.bottleneck = bottleneck
        target_module.adapter_layer = Houlsby_Adapter(
            input_size=input_size, bottleneck=bottleneck
        )
        adapter_added_layer = target_module

    elif isinstance(target_module, TransformerSentenceEncoderLayer):

        if HoulsbyTransformerSentenceEncoderLayer is None:
            raise ImportError(
                "Error: S3PRL is not properly installed."
                "Please install S3PRL: cd ${MAIN_ROOT}/tools && make s3prl.done"
            )

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
            if "adapter" in n:
                continue
            p.requires_grad = eval(f"target_module.{n}").requires_grad

        # copy weights from the target module
        orig_state_dict = target_module.state_dict()
        adapter_added_layer.load_state_dict(orig_state_dict, strict=False)

        # Copy all hooks to the new layer
        for k, v in target_module.__dict__.items():
            if "hook" not in k:
                continue
            adapter_added_layer.__dict__[k] = v
    else:
        raise NotImplementedError(
            f"Target module {type(target_module)} is not supported."
        )
    return adapter_added_layer


@typechecked
def create_new_lora_module(
    target_module: torch.nn.Module,
    rank: int,
    alpha: int,
    dropout_rate: float,
    adapter_type: str = "lora",
    **backend_kwargs,
):
    """Create a new LoRA-family module for the given target module."""
    bias = hasattr(target_module, "bias") and target_module.bias is not None

    if isinstance(target_module, torch.nn.Embedding):
        return LoraEmbedding(
            target_module.num_embeddings,
            target_module.embedding_dim,
            r=rank,
            lora_alpha=alpha,
        )

    if isinstance(target_module, torch.nn.Linear):
        adapter_type_lower = adapter_type.lower()
        if adapter_type_lower not in LINEAR_BACKENDS:
            raise ValueError(
                f"Unsupported adapter_type '{adapter_type}' for torch.nn.Linear. "
                f"Available types: {sorted(LINEAR_BACKENDS.keys())}."
            )
        backend_cls = LINEAR_BACKENDS[adapter_type_lower]
        return backend_cls(
            target_module.in_features,
            target_module.out_features,
            bias=bias,
            r=rank,
            lora_alpha=alpha,
            lora_dropout=dropout_rate,
            **backend_kwargs,
        )

    raise ValueError(
        f"Target module {target_module} is not supported. "
        f"Currently, only `torch.nn.Embedding` and `torch.nn.Linear` are supported."
    )
