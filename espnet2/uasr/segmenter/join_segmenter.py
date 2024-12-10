import argparse
from typing import Dict, Optional

import torch
from typeguard import typechecked

from espnet2.uasr.segmenter.abs_segmenter import AbsSegmenter
from espnet2.utils.types import str2bool


class JoinSegmenter(AbsSegmenter):
    """
        JoinSegmenter is a segmenter that processes input tensors to produce
    segmented outputs based on a join strategy. It is designed to work
    within the ESPnet framework for unsupervised audio segmentation.

    Attributes:
        subsampling_rate (float): The rate at which to subsample the input.
        mean_pool (bool): Whether to use mean pooling on the logits.
        mean_pool_join (bool): Whether to use mean pooling on the joined segments.
        remove_zeros (bool): Whether to remove segments that are zero-valued.

    Args:
        cfg (Optional[Dict]): Configuration dictionary for the segmenter. If provided,
            it must contain a 'segmentation' key with a subkey 'type' set to 'JOIN'.
        subsample_rate (float): The rate at which to subsample the input. Default is 0.25.
        mean_pool (str2bool): Flag to enable mean pooling. Default is True.
        mean_join_pool (str2bool): Flag to enable mean pooling on joined segments.
            Default is False.
        remove_zeros (str2bool): Flag to remove zero-valued segments. Default is False.

    Methods:
        pre_segment(xs_pad: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
            Prepares the input tensor and padding mask for segmentation.

        logit_segment(logits: torch.Tensor, padding_mask: torch.Tensor) -> torch.Tensor:
            Processes the logits and padding mask to produce the segmented outputs.

    Examples:
        # Initialize the JoinSegmenter with default parameters
        segmenter = JoinSegmenter()

        # Pre-segmenting inputs
        xs_pad, padding_mask = segmenter.pre_segment(xs_pad_tensor, padding_mask_tensor)

        # Getting the segmented logits
        segmented_logits, new_padding_mask = segmenter.logit_segment(logits_tensor, padding_mask_tensor)

    Note:
        This segmenter assumes that the input logits are the output of a model's
        prediction layer.

    Todo:
        - Add more configurable options for the segmentation strategy.
    """

    @typechecked
    def __init__(
        self,
        cfg: Optional[Dict] = None,
        subsample_rate: float = 0.25,
        mean_pool: str2bool = True,
        mean_join_pool: str2bool = False,
        remove_zeros: str2bool = False,
    ):
        super().__init__()

        if cfg is not None:
            cfg = argparse.Namespace(**cfg["segmentation"])
            assert cfg.type == "JOIN"
            self.subsampling_rate = cfg.subsample_rate
            self.mean_pool = cfg.mean_pool
            self.mean_pool_join = cfg.mean_pool_join
            self.remove_zeros = cfg.remove_zeros
        else:
            self.mean_pool_join = mean_join_pool
            self.remove_zeros = remove_zeros

    @typechecked
    def pre_segment(
        self,
        xs_pad: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
                Preprocesses input tensors for segmentation by returning the original tensors.

        This method serves as a preliminary step before applying the segmentation logic.
        It currently returns the input tensors without modification. This is useful for
        ensuring compatibility with subsequent methods in the segmentation process.

        Args:
            xs_pad (torch.Tensor): A tensor representing the input features to be
                processed. This tensor can have any shape that is compatible with
                the segmentation model.
            padding_mask (torch.Tensor): A tensor indicating the padding in the
                input features. It should have the same shape as `xs_pad`, with
                boolean values indicating padded positions.

        Returns:
            torch.Tensor: The original input tensor `xs_pad` and the `padding_mask`
                without any modifications.

        Examples:
            >>> import torch
            >>> segmenter = JoinSegmenter()
            >>> xs_pad = torch.randn(2, 5, 10)  # Example input tensor
            >>> padding_mask = torch.tensor([[False, False, True, True, True],
            ...                               [False, False, False, True, True]])
            >>> processed_xs, processed_mask = segmenter.pre_segment(xs_pad, padding_mask)
            >>> assert torch.equal(xs_pad, processed_xs)
            >>> assert torch.equal(padding_mask, processed_mask)
        """
        return xs_pad, padding_mask

    @typechecked
    def logit_segment(
        self,
        logits: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
                Logit segment function that processes logits and applies padding masks to segment
        the output.

        This method takes the logits and a padding mask as input, then identifies the
        predicted classes based on the highest logits. It handles padding by marking
        the corresponding predictions as -1. The function also manages the removal of
        zero entries based on the configuration and adjusts the logits accordingly.

        Attributes:
            remove_zeros (bool): Flag indicating whether to remove zero entries from
                the logits.

        Args:
            logits (torch.Tensor): A tensor of shape (batch_size, time_length,
                channel_size) containing the model's raw output logits.
            padding_mask (torch.Tensor): A tensor of shape (batch_size, time_length)
                that indicates the padding positions in the logits.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, new_time_length,
                channel_size) containing the processed logits after segmentation and
                padding adjustments.

        Examples:
            >>> logits = torch.randn(2, 5, 3)  # Example logits
            >>> padding_mask = torch.tensor([[0, 0, 0, 1, 1], [0, 0, 1, 1, 1]])  # Example mask
            >>> segmenter = JoinSegmenter()
            >>> new_logits, new_pad = segmenter.logit_segment(logits, padding_mask)

        Note:
            This method is designed to work within the context of a JoinSegmenter
            instance and assumes that the instance has been properly initialized
            with configuration parameters.
        """
        preds = logits.argmax(dim=-1)

        if padding_mask.any():
            preds[padding_mask] = -1  # mark pad
        uniques = []

        batch_size, time_length, channel_size = logits.shape

        for p in preds:
            uniques.append(
                p.cpu().unique_consecutive(return_inverse=True, return_counts=True)
            )

        new_time_length = max(u[0].numel() for u in uniques)
        new_logits = logits.new_zeros(batch_size, new_time_length, channel_size)
        new_pad = padding_mask.new_zeros(batch_size, new_time_length)

        for b in range(batch_size):
            value, index, count = uniques[b]
            keep = value != -1

            if self.remove_zeros:
                keep.logical_and_(value != 0)

            if self.training and not self.mean_pool_join:
                value[0] = 0
                value[1:] = count.cumsum(0)[:-1]
                part = count > 1
                random = torch.rand(part.sum())
                value[part] += (count[part] * random).long()
                new_logits[b, : value.numel()] = logits[b, value]
            else:
                new_logits[b].index_add_(
                    dim=0, index=index.to(new_logits.device), source=logits[b]
                )
                new_logits[b, : count.numel()] = new_logits[
                    b, : count.numel()
                ] / count.unsqueeze(-1).to(new_logits.device)

            new_size = keep.sum()
            if not keep.all():
                kept_logits = new_logits[b, : count.numel()][keep]
                new_logits[b, :new_size] = kept_logits

            if new_size < new_time_length:
                pad = new_time_length - new_size
                new_logits[b, -pad:] = 0
                new_pad[b, -pad:] = True
        return new_logits, new_pad
