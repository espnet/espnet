from collections import defaultdict
from typing import Dict
from typing import List

import torch

from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention
from espnet2.layers.abs_attention import AbsAttention
from espnet2.train.abs_espnet_model import AbsESPnetModel


@torch.no_grad()
def calculate_all_attentions(
    model: AbsESPnetModel, batch: Dict[str, torch.Tensor]
) -> Dict[str, List[torch.Tensor]]:
    """Derive the outputs from the all attention layers

    Args:
        model:
        batch: same as forward
    Returns:
        return_dict: A dict of a list of tensor.
        key_names x batch x (D1, D2, ...)

    """
    bs = len(next(iter(batch.values())))
    assert all(len(v) == bs for v in batch.values()), {
        k: v.shape for k, v in batch.items()
    }

    # 1. Register forward_hook fn to save the output from specific layers
    outputs = {}
    handles = {}
    for name, modu in model.named_modules():

        def hook(module, input, output, name=name):
            # TODO(kamo): Should unify the interface?
            if isinstance(module, MultiHeadedAttention):
                outputs[name] = output
            else:
                c, w = output
                outputs[name] = w

        if isinstance(modu, AbsAttention):
            handle = modu.register_forward_hook(hook)
            handles[name] = handle

    # 2. Just forward one by one sample.
    # Batch-mode can't be used to keep requirements small for each models.
    keys = []
    for k in batch:
        if not k.endswith("_lengths"):
            keys.append(k)

    return_dict = defaultdict(list)
    for ibatch in range(bs):
        # *: (B, L, ...) -> (1, L2, ...)
        _sample = {
            k: batch[k][ibatch, None, : batch[k + "_lengths"][ibatch]]
            if k + "_lengths" in batch
            else batch[k][ibatch, None]
            for k in keys
        }

        # *_lengths: (B,) -> (1,)
        _sample.update(
            {
                k + "_lengths": batch[k + "_lengths"][ibatch, None]
                for k in keys
                if k + "_lengths" in batch
            }
        )
        model(**_sample)

        # Derive the attention results
        for name, output in outputs.items():
            return_dict[name].append(output)

    # 3. Remove all hooks
    for _, handle in handles.items():
        handle.remove()

    return dict(return_dict)
