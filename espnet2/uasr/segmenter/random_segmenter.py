import math

import torch
from typeguard import typechecked

from espnet2.uasr.segmenter.abs_segmenter import AbsSegmenter
from espnet2.utils.types import str2bool


class RandomSegmenter(AbsSegmenter):
    """
        RandomSegmenter is a class that segments input data randomly based on a
    subsampling rate. It inherits from the AbsSegmenter class and provides
    methods for pre-segmenting input data and generating logit segments.

    Attributes:
        subsample_rate (float): The rate at which to subsample the input data.
            Default is 0.25.

    Args:
        subsample_rate (float): The proportion of the input to retain during
            segmentation. Defaults to 0.25.
        mean_pool (str2bool): Whether to apply mean pooling. Defaults to True.
        mean_join_pool (str2bool): Whether to apply mean join pooling.
            Defaults to False.
        remove_zeros (str2bool): Whether to remove zero entries from the
            segments. Defaults to False.

    Methods:
        pre_segment(xs_pad: torch.Tensor, padding_mask: torch.Tensor) ->
            torch.Tensor:
            Segments the input tensor by randomly selecting a subset based on the
            subsample rate.

        logit_segment(xs_pad: torch.Tensor, padding_mask: torch.Tensor) ->
            torch.Tensor:
            Returns the input tensor and padding mask without modification.

    Examples:
        segmenter = RandomSegmenter(subsample_rate=0.5)
        xs_pad = torch.rand(10, 100, 20)  # Batch of 10, 100 time steps, 20 features
        padding_mask = torch.ones(10, 100)

        segmented_data, new_padding_mask = segmenter.pre_segment(xs_pad,
                                                                 padding_mask)
        logits = segmenter.logit_segment(xs_pad, padding_mask)

    Note:
        This segmenter does not require any learned parameters and relies solely
        on the random selection of indices.

    Todo:
        Implement additional segmentation strategies in future versions.
    """

    @typechecked
    def __init__(
        self,
        subsample_rate: float = 0.25,
        mean_pool: str2bool = True,
        mean_join_pool: str2bool = False,
        remove_zeros: str2bool = False,
    ):
        super().__init__()
        self.subsample_rate = subsample_rate

    def pre_segment(
        self,
        xs_pad: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
                Prepares the input tensor and padding mask for segmentation by randomly
        subsampling the input data based on the specified subsample rate.

        This method takes a padded input tensor and its corresponding padding mask
        and reduces the input tensor's length to a target number of elements, which
        is calculated using the subsample rate. The output is a new tensor and
        mask that reflect this subsampling.

        Args:
            xs_pad (torch.Tensor): The padded input tensor with shape
                (batch_size, sequence_length, features).
            padding_mask (torch.Tensor): The mask tensor indicating valid input
                positions, with shape (batch_size, sequence_length).

        Returns:
            torch.Tensor: A tuple containing the subsampled input tensor and the
                updated padding mask.

        Examples:
            >>> segmenter = RandomSegmenter(subsample_rate=0.5)
            >>> xs_pad = torch.randn(2, 10, 5)  # Example input tensor
            >>> padding_mask = torch.ones(2, 10)  # Example padding mask
            >>> subsampled_tensor, updated_mask = segmenter.pre_segment(xs_pad,
            ...                                                          padding_mask)
            >>> print(subsampled_tensor.shape)  # Output shape will depend on
            ...                                   # the subsample rate
        """
        target_num = math.ceil(xs_pad.size(1) * self.subsample_rate)
        ones = torch.ones(xs_pad.shape[:-1], device=xs_pad.device)
        indices, _ = ones.multinomial(target_num).sort(dim=-1)
        indices_ld = indices.unsqueeze(-1).expand(-1, -1, xs_pad.size(-1))
        xs_pad = xs_pad.gather(1, indices_ld)
        padding_mask = padding_mask.gather(1, index=indices)
        return xs_pad, padding_mask

    def logit_segment(
        self,
        xs_pad: torch.Tensor,
        padding_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
            Computes the segment logits based on the padded input tensor and padding mask.

        This method processes the input tensor `xs_pad` and the associated
        `padding_mask`, returning the logits for each segment. The logits can
        be utilized in subsequent steps for segment classification or other
        tasks within the segmentation framework.

        Args:
            xs_pad (torch.Tensor): A padded input tensor of shape (batch_size,
                seq_length, features) that represents the input data.
            padding_mask (torch.Tensor): A tensor of shape (batch_size, seq_length)
                that indicates the valid entries in `xs_pad`. It is typically a
                binary mask where 1 indicates a valid token and 0 indicates padding.

        Returns:
            torch.Tensor: The segment logits, which is the same shape as `xs_pad`
                and `padding_mask`. The output tensor can be used for further
                processing in segmentation tasks.

        Examples:
            >>> xs_pad = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0],
            ... [0.0, 0.0]]])
            >>> padding_mask = torch.tensor([[1, 1], [1, 0]])
            >>> segmenter = RandomSegmenter()
            >>> logits = segmenter.logit_segment(xs_pad, padding_mask)
            >>> print(logits)
            (tensor([[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [0.0, 0.0]]]),
             tensor([[1, 1], [1, 0]]))
        """
        return xs_pad, padding_mask
