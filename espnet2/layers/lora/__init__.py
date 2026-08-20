"""Self-contained LoRA / PEFT backends used by ESPnet."""

from espnet2.layers.lora.layers import (
    DoraLinear,
    Embedding,
    Linear,
    LoRALayer,
    PiSSALinear,
    SSVDLinear,
    SVFTLinear,
)
from espnet2.layers.lora.utils import lora_state_dict, mark_only_lora_as_trainable

LINEAR_BACKENDS = {
    "lora": Linear,
    "dora": DoraLinear,
    "pissa": PiSSALinear,
    "svft": SVFTLinear,
    "ssvd": SSVDLinear,
}

__all__ = [
    "DoraLinear",
    "Embedding",
    "LINEAR_BACKENDS",
    "Linear",
    "LoRALayer",
    "PiSSALinear",
    "SSVDLinear",
    "SVFTLinear",
    "lora_state_dict",
    "mark_only_lora_as_trainable",
]
