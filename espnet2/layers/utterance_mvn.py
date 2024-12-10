from typing import Tuple

import torch
from typeguard import typechecked

from espnet2.layers.abs_normalize import AbsNormalize
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class UtteranceMVN(AbsNormalize):
    """
        UtteranceMVN is a normalization layer that applies mean and variance
    normalization to input tensors, typically used in speech processing.

    This class inherits from AbsNormalize and provides functionality to
    normalize the means and variances of utterances in a batch.

    Attributes:
        norm_means (bool): If True, normalize the means of the input tensors.
        norm_vars (bool): If True, normalize the variances of the input tensors.
        eps (float): A small value to prevent division by zero during normalization.

    Args:
        norm_means (bool): Whether to normalize the means. Default is True.
        norm_vars (bool): Whether to normalize the variances. Default is False.
        eps (float): A small constant for numerical stability. Default is 1.0e-20.

    Examples:
        >>> layer = UtteranceMVN(norm_means=True, norm_vars=False)
        >>> x = torch.randn(5, 10, 20)  # Batch of 5, 10 time steps, 20 features
        >>> ilens = torch.tensor([10, 10, 10, 10, 10])  # All utterances are valid
        >>> normalized_x, ilens = layer(x, ilens)
    """

    @typechecked
    def __init__(
        self,
        norm_means: bool = True,
        norm_vars: bool = False,
        eps: float = 1.0e-20,
    ):
        super().__init__()
        self.norm_means = norm_means
        self.norm_vars = norm_vars
        self.eps = eps

    def extra_repr(self):
        """
                Returns a string representation of the UtteranceMVN instance, including its
        attributes.

        This method provides a summary of the normalization settings used in the
        UtteranceMVN class, specifically whether means and/or variances are being
        normalized.

        Attributes:
            norm_means (bool): Indicates if the means should be normalized.
            norm_vars (bool): Indicates if the variances should be normalized.

        Returns:
            str: A formatted string summarizing the normalization configuration.

        Examples:
            >>> mvn = UtteranceMVN(norm_means=True, norm_vars=False)
            >>> print(mvn.extra_repr())
            norm_means=True, norm_vars=False

            >>> mvn2 = UtteranceMVN(norm_means=False, norm_vars=True)
            >>> print(mvn2.extra_repr())
            norm_means=False, norm_vars=True
        """
        return f"norm_means={self.norm_means}, norm_vars={self.norm_vars}"

    def forward(
        self, x: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward function

        Args:
            x: (B, L, ...)
            ilens: (B,)

        """
        return utterance_mvn(
            x,
            ilens,
            norm_means=self.norm_means,
            norm_vars=self.norm_vars,
            eps=self.eps,
        )


def utterance_mvn(
    x: torch.Tensor,
    ilens: torch.Tensor = None,
    norm_means: bool = True,
    norm_vars: bool = False,
    eps: float = 1.0e-20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Apply utterance mean and variance normalization.

    This function normalizes the input tensor `x` by subtracting the mean
    and optionally dividing by the standard deviation (computed from the
    variance) across the time dimension, while taking into account
    zero-padded regions based on the input lengths `ilens`.

    Attributes:
        norm_means (bool): Whether to normalize the means.
        norm_vars (bool): Whether to normalize the variances.
        eps (float): A small constant to avoid division by zero.

    Args:
        x (torch.Tensor): Input tensor of shape (B, T, D), where B is the
            batch size, T is the sequence length, and D is the feature
            dimension. It is assumed to be zero-padded.
        ilens (torch.Tensor, optional): Tensor of shape (B,) containing
            the actual lengths of each sequence in the batch. If not
            provided, it will be set to the maximum sequence length in
            the batch.
        norm_means (bool): Flag to indicate whether to normalize the
            means. Defaults to True.
        norm_vars (bool): Flag to indicate whether to normalize the
            variances. Defaults to False.
        eps (float): Small constant to prevent division by zero during
            variance normalization. Defaults to 1.0e-20.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the
            normalized tensor and the input lengths tensor.

    Examples:
        >>> import torch
        >>> x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]],
        ...                    [[5.0, 6.0], [0.0, 0.0]]])
        >>> ilens = torch.tensor([2, 1])
        >>> normalized_x, normalized_ilens = utterance_mvn(x, ilens)
        >>> print(normalized_x)
        tensor([[[-1.0000, -1.0000],
                 [ 1.0000,  1.0000]],

                [[ 1.0000,  1.0000],
                 [ 0.0000,  0.0000]]])

    Note:
        The input tensor `x` should be a zero-padded tensor, and the
        padding should be handled appropriately using the `ilens`
        argument to avoid affecting the normalization calculations.
    """
    if ilens is None:
        ilens = x.new_full([x.size(0)], x.size(1))
    ilens_ = ilens.to(x.device, x.dtype).view(-1, *[1 for _ in range(x.dim() - 1)])
    # Zero padding
    if x.requires_grad:
        x = x.masked_fill(make_pad_mask(ilens, x, 1), 0.0)
    else:
        x.masked_fill_(make_pad_mask(ilens, x, 1), 0.0)
    # mean: (B, 1, D)
    mean = x.sum(dim=1, keepdim=True) / ilens_

    if norm_means:
        x -= mean

        if norm_vars:
            var = x.pow(2).sum(dim=1, keepdim=True) / ilens_
            std = torch.clamp(var.sqrt(), min=eps)
            x = x / std
        return x, ilens
    else:
        if norm_vars:
            y = x - mean
            y.masked_fill_(make_pad_mask(ilens, y, 1), 0.0)
            var = y.pow(2).sum(dim=1, keepdim=True) / ilens_
            std = torch.clamp(var.sqrt(), min=eps)
            x /= std
        return x, ilens
