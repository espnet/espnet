"""Definition of the low-rank adaptation (LoRA) for large models.

References:
    1. LoRA: Low-Rank Adaptation of Large Language Models
       (https://arxiv.org/pdf/2106.09685.pdf)
    2. https://github.com/microsoft/LoRA.git
    3. https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora.py

"""

import torch
from typeguard import typechecked

from espnet2.layers.create_adapter_fn import create_houlsby_adapter, create_lora_adapter

create_adapter_fn_table = {
    "lora": create_lora_adapter,
    "houlsby": create_houlsby_adapter,
}


@typechecked
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
    assert adapter in create_adapter_fn_table, f"Adapter {adapter} is not supported."
    create_adapter_fn = create_adapter_fn_table[adapter]
    create_adapter_fn(model=model, **adapter_conf)
