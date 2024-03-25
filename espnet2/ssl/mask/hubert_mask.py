# https://github.com/pytorch/audio/blob/main/src/torchaudio/models/wav2vec2/components.py#L972

from typing import Dict, List, Optional, Tuple, Union

import torch
from torch import Tensor, nn

from espnet2.ssl.mask.abs_mask import AbsMasker

class HubertMasker(AbsMasker):
    """Generate the masks for masked prediction.
    Args:
        encoder_embed_dim (int): The dimension of the transformer embedding output.
        mask_prob (float): Prob for each token to be the start of a masked span.
            This will be multiplied by num of timesteps divided by len of mask span to mask
            approx this % of all elements. However due to overlaps, the actual number
            will be smaller (unless no_overlap is True).
        mask_selection (str): How to choose the mask length.
            Options: [``static``, ``uniform``, ``normal``, ``poisson``].
        mask_other (float): Secondary mask argument (used for more complex distributions).
        mask_length (int): The lengths of the mask.
        no_mask_overlap (bool):  Whether to allow masks to overlap.
        mask_min_space (int):  Minimum space between spans (if no overlap is enabled).
        mask_channel_prob (float): The probability of replacing a feature with 0.
        mask_channel_selection (str): How to choose the mask length for channel masking.
            Options: [``static``, ``uniform``, ``normal``, ``poisson``].
        mask_channel_other (float): Secondary mask argument for channel masking
            (used for more complex distributions).
        mask_channel_length (int): Minimum space between spans 
            (if no overlap is enabled) for channel masking.
        no_mask_channel_overlap (bool):  Whether to allow channel masks to overlap.
        mask_channel_min_space (int): Minimum space between spans for channel masking
            (if no overlap is enabled).
    """

    def __init__(
        self,
        encoder_embed_dim: int,
        mask_prob: float,
        mask_selection: str,
        mask_other: float,
        mask_length: int,
        no_mask_overlap: bool,
        mask_min_space: int,
        mask_channel_prob: float,
        mask_channel_selection: str,
        mask_channel_other: float,
        mask_channel_length: int,
        no_mask_channel_overlap: bool,
        mask_channel_min_space: int,
    ):
        super().__init__()
        self.mask_prob = mask_prob
        self.mask_selection = mask_selection
        self.mask_other = mask_other
        self.mask_length = mask_length
        self.no_mask_overlap = no_mask_overlap
        self.mask_min_space = mask_min_space
        self.mask_channel_prob = mask_channel_prob
        self.mask_channel_selection = mask_channel_selection
        self.mask_channel_other = mask_channel_other
        self.mask_channel_length = mask_channel_length
        self.no_mask_channel_overlap = no_mask_channel_overlap
        self.mask_channel_min_space = mask_channel_min_space
        self.mask_embedding = nn.Parameter(torch.FloatTensor(encoder_embed_dim))
        torch.nn.init.uniform_(self.mask_embedding)

    def forward(self, x: Tensor, padding_mask: Optional[Tensor]) -> Tensor:
        """
        Args:
            x (Tensor): The encoded representations after feature extraction module.
            padding_mask (Tensor or None): The padding mask
                which will prevent masking padded elements.

        Returns:
            Tensor: The feature representations after masking.
            Tensor: The generated mask indices.
        """
        B, T, C = x.shape
        if self.mask_prob > 0:

            mask_indices = _compute_mask_indices(
                (B, T),
                padding_mask,
                x.device,
                self.mask_prob,
                self.mask_length,
                self.mask_selection,
                self.mask_other,
                min_masks=2,
                no_overlap=self.no_mask_overlap,
                min_space=self.mask_min_space,
            )
            # change dtype of mask_embedding to x for mixed-precision training.
            # see https://github.com/pytorch/audio/issues/2847 for details.
            x[mask_indices] = self.mask_embedding.to(x.dtype)
        else:
            mask_indices = None

        if self.mask_channel_prob > 0:
            mask_channel_indices = _compute_mask_indices(
                (B, C),
                None,
                self.mask_channel_prob,
                self.mask_channel_length,
                self.mask_channel_selection,
                self.mask_channel_other,
                no_overlap=self.no_mask_channel_overlap,
                min_space=self.mask_channel_min_space,
            )
            mask_channel_indices = (
                mask_channel_indices.to(x.device).unsqueeze(1).expand(-1, T, -1)
            )
            x[mask_channel_indices] = 0

        return x, {"mask_m": mask_indices, "mask_u": ~mask_indices}


def _compute_mask_indices(
    shape: Tuple[int, int],
    padding_mask: Optional[Tensor],
    device: str,
    mask_prob: float,
    mask_length: int,
    mask_type: str = "static",
    mask_other: float = 0.0,
    min_masks: int = 0,
    no_overlap: bool = False,
    min_space: int = 0,
) -> Tensor:
    """Computes random mask spans for a given shape.
    Args:
        shape (int, int): The shape for which to compute masks.
            The first element is batch size and second is the number of frames.
        padding_mask (Tensor or None): The padding mask of the same dimension as shape,
            which will prevent masking padded elements.
        mask_prob (float): Prob to be chosen as start of the span to be masked.
            Will be multiplied by num timesteps divided by length of mask span to mask
            approx this percentage of all elements. Due to overlaps, the actual number
            will be smaller (unless no_overlap is True).
        mask_type (str): How to compute mask lengths.
            ``static``: Fixed size
            ``uniform``: Sample from uniform distribution [mask_other, mask_length*2]
            ``normal``: Sample from normal dist with 
                mean ``mask_length`` and stdev ``mask_other``.
            ``poisson``: Sample from possion distribution with lambda = ``mask_length``.
        min_masks (int): Minimum number of masked spans.
        no_overlap (bool): If false, will switch to an alternative recursive algorithm
            that prevents spans from overlapping.
        min_space (int): How many frames to keep unmasked between spans 
            (if no_overlap is True).

    Returns:
        (Tensor): The mask indices of dimension `[batch, frame]`.
    """

    batch_size, frame = shape
    mask = torch.full((batch_size, frame), False)
    # add a random number for probabilistic rounding
    all_num_mask = int(mask_prob * frame / float(mask_length) + torch.rand(1))

    all_num_mask = max(min_masks, all_num_mask)

    mask_idcs = []
    for i in range(batch_size):
        if padding_mask is not None:
            sz = frame - padding_mask[i].long().sum().item()
            # add a random number for probabilistic rounding
            num_mask = int(mask_prob * sz / float(mask_length) + torch.rand(1))
            num_mask = max(min_masks, num_mask)
        else:
            sz = frame
            num_mask = all_num_mask

        if mask_type == "static":
            lengths = torch.full((num_mask,), mask_length)
        elif mask_type == "uniform":
            lengths = torch.randint(mask_other, mask_length * 2 + 1, size=(num_mask,))
        elif mask_type == "normal":
            lengths = torch.normal(mask_length, mask_other, size=(num_mask,))
            lengths = torch.maximum(torch.ones(1), torch.round(lengths)).int()
        elif mask_type == "poisson":
            lengths = torch.poisson(mask_length, size=(num_mask,))
            lengths = torch.round(lengths).int()
        else:
            raise Exception(f"unknown mask selection: {mask_type}")

        if sum(lengths) == 0:
            lengths[0] = min(mask_length, sz - 1)

        if no_overlap:
            mask_idc = []

            def arrange(s, e, length, keep_length):
                span_start = torch.randint(s, e - length, size=(1,))
                mask_idc.extend(span_start + i for i in range(length))

                new_parts = []
                if span_start - s - min_space >= keep_length:
                    new_parts.append((s, span_start - min_space + 1))
                if e - span_start - keep_length - min_space > keep_length:
                    new_parts.append((span_start + length + min_space, e))
                return new_parts

            parts = [(0, sz)]
            min_length = min(lengths)
            for length in sorted(lengths, reverse=True):
                lens = torch.tensor([e - s for s, e in parts], dtype=torch.int)
                lens[lens < length + min_space] = 0
                l_sum = lens.sum()
                if l_sum == 0:
                    break
                probs = lens / l_sum
                c = torch.distributions.categorical.Categorical(probs).sample()
                s, e = parts.pop(c)
                parts.extend(arrange(s, e, length, min_length))
            mask_idc = torch.tensor(mask_idc)
        else:
            min_len = min(lengths)
            if sz - min_len <= num_mask:
                min_len = sz - num_mask - 1

            mask_idc = torch.randperm(sz - min_len)[:num_mask]
            mask_idc = torch.tensor(
                [
                    mask_idc[j] + offset
                    for j in range(len(mask_idc))
                    for offset in range(lengths[j])
                ]
            )

        mask_idcs.append(torch.unique(mask_idc[mask_idc < sz]))

    min_len = min([len(m) for m in mask_idcs])
    for i, mask_idc in enumerate(mask_idcs):
        if len(mask_idc) > min_len:
            mask_idc = mask_idc[torch.randperm(len(mask_idc))[:min_len].long()]
        mask[i, mask_idc] = True

    return mask
