from collections import defaultdict
from typing import Dict, List

import torch

from espnet2.gan_tts.jets.alignments import AlignmentModule
from espnet2.train.abs_espnet_model import AbsESPnetModel
from espnet.nets.pytorch_backend.rnn.attentions import (
    AttAdd,
    AttCov,
    AttCovLoc,
    AttDot,
    AttForward,
    AttForwardTA,
    AttLoc,
    AttLoc2D,
    AttLocRec,
    AttMultiHeadAdd,
    AttMultiHeadDot,
    AttMultiHeadLoc,
    AttMultiHeadMultiResLoc,
    NoAtt,
)
from espnet.nets.pytorch_backend.transformer.attention import MultiHeadedAttention


@torch.no_grad()
def calculate_all_attentions(
    model: AbsESPnetModel, batch: Dict[str, torch.Tensor]
) -> Dict[str, List[torch.Tensor]]:
    """
        Derive the outputs from all attention layers in a given model.

    This function registers forward hooks on the attention layers of the provided
    model and collects the attention outputs during a forward pass using a given
    batch of data. The collected attention outputs are returned in a structured
    dictionary format.

    Args:
        model (AbsESPnetModel): The ESPnet model containing attention layers.
        batch (Dict[str, torch.Tensor]): A dictionary of input tensors for the
            model's forward pass. It should include all necessary input features
            and lengths.

    Returns:
        Dict[str, List[torch.Tensor]]: A dictionary where keys are the names of
            the attention layers and values are lists of tensors representing the
            attention outputs. The shape of each tensor will depend on the specific
            attention mechanism, typically in the form of key_names x batch x (D1,
            D2, ...).

    Examples:
        >>> model = MyESPnetModel()
        >>> batch = {
        ...     'input': torch.randn(2, 10, 20),
        ...     'input_lengths': torch.tensor([10, 10]),
        ... }
        >>> attentions = calculate_all_attentions(model, batch)
        >>> print(attentions.keys())
        dict_keys(['layer1_att', 'layer2_att', ...])

    Note:
        The function runs in evaluation mode (with `torch.no_grad()`) to prevent
        gradients from being calculated, which can improve performance during
        inference.
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
            if isinstance(module, MultiHeadedAttention):
                # NOTE(kamo): MultiHeadedAttention doesn't return attention weight
                # attn: (B, Head, Tout, Tin)
                outputs[name] = module.attn.detach().cpu()
            elif isinstance(module, AttLoc2D):
                c, w = output
                # w: previous concate attentions
                # w: (B, nprev, Tin)
                att_w = w[:, -1].detach().cpu()
                outputs.setdefault(name, []).append(att_w)
            elif isinstance(module, (AttCov, AttCovLoc)):
                c, w = output
                assert isinstance(w, list), type(w)
                # w: list of previous attentions
                # w: nprev x (B, Tin)
                att_w = w[-1].detach().cpu()
                outputs.setdefault(name, []).append(att_w)
            elif isinstance(module, AttLocRec):
                # w: (B, Tin)
                c, (w, (att_h, att_c)) = output
                att_w = w.detach().cpu()
                outputs.setdefault(name, []).append(att_w)
            elif isinstance(
                module,
                (
                    AttMultiHeadDot,
                    AttMultiHeadAdd,
                    AttMultiHeadLoc,
                    AttMultiHeadMultiResLoc,
                ),
            ):
                c, w = output
                # w: nhead x (B, Tin)
                assert isinstance(w, list), type(w)
                att_w = [_w.detach().cpu() for _w in w]
                outputs.setdefault(name, []).append(att_w)
            elif isinstance(
                module,
                (
                    AttAdd,
                    AttDot,
                    AttForward,
                    AttForwardTA,
                    AttLoc,
                    NoAtt,
                ),
            ):
                c, w = output
                att_w = w.detach().cpu()
                outputs.setdefault(name, []).append(att_w)
            elif isinstance(module, AlignmentModule):
                w = output
                att_w = torch.exp(w).detach().cpu()
                outputs.setdefault(name, []).append(att_w)

        handle = modu.register_forward_hook(hook)
        handles[name] = handle

    # 2. Just forward one by one sample.
    # Batch-mode can't be used to keep requirements small for each models.
    keys = []
    for k in batch:
        if not (k.endswith("_lengths") or k in ["utt_id"]):
            keys.append(k)

    return_dict = defaultdict(list)
    for ibatch in range(bs):
        # *: (B, L, ...) -> (1, L2, ...)
        _sample = {
            k: (
                batch[k][ibatch, None, : batch[k + "_lengths"][ibatch]]
                if k + "_lengths" in batch
                else batch[k][ibatch, None]
            )
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

        if "utt_id" in batch:
            _sample["utt_id"] = batch["utt_id"]

        model(**_sample)

        # Derive the attention results
        for name, output in outputs.items():
            if isinstance(output, list):
                if isinstance(output[0], list):
                    # output: nhead x (Tout, Tin)
                    output = torch.stack(
                        [
                            # Tout x (1, Tin) -> (Tout, Tin)
                            torch.cat([o[idx] for o in output], dim=0)
                            for idx in range(len(output[0]))
                        ],
                        dim=0,
                    )
                else:
                    # Tout x (1, Tin) -> (Tout, Tin)
                    output = torch.cat(output, dim=0)
            else:
                # output: (1, NHead, Tout, Tin) -> (NHead, Tout, Tin)
                output = output.squeeze(0)
            # output: (Tout, Tin) or (NHead, Tout, Tin)
            return_dict[name].append(output)
        outputs.clear()

    # 3. Remove all hooks
    for _, handle in handles.items():
        handle.remove()

    return dict(return_dict)
