from pathlib import Path
from typing import Tuple, Union

import numpy as np
import torch
from typeguard import typechecked

from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.layers.inversible_interface import InversibleInterface
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask


class GlobalMVN(AbsNormalize, InversibleInterface):
    """
        Apply global mean and variance normalization.

    This class implements global mean and variance normalization, which can be
    used to normalize features by their mean and variance across a dataset. It
    can read the statistics from a specified `.npy` or `.npz` file, and
    normalization can be applied based on the specified parameters.

    TODO(kamo): Make this class portable somehow.

    Attributes:
        stats_file (Path): Path to the statistics file containing mean and variance.
        norm_means (bool): Whether to apply mean normalization. Defaults to True.
        norm_vars (bool): Whether to apply variance normalization. Defaults to True.
        eps (float): Small constant to prevent division by zero. Defaults to 1.0e-20.
        mean (torch.Tensor): Mean values loaded from the statistics file.
        std (torch.Tensor): Standard deviation values loaded from the statistics file.

    Args:
        stats_file: npy file containing statistics for normalization.
        norm_means: Apply mean normalization (default: True).
        norm_vars: Apply variance normalization (default: True).
        eps: Small constant to prevent division by zero (default: 1.0e-20).

    Examples:
        >>> global_mvn = GlobalMVN("path/to/stats_file.npy")
        >>> normalized_tensor, ilens = global_mvn(input_tensor, input_lengths)

        >>> denormalized_tensor, ilens = global_mvn.inverse(normalized_tensor, ilens)
    """

    @typechecked
    def __init__(
        self,
        stats_file: Union[Path, str],
        norm_means: bool = True,
        norm_vars: bool = True,
        eps: float = 1.0e-20,
    ):
        super().__init__()
        self.norm_means = norm_means
        self.norm_vars = norm_vars
        self.eps = eps
        stats_file = Path(stats_file)

        self.stats_file = stats_file
        stats = np.load(stats_file)
        if isinstance(stats, np.ndarray):
            # Kaldi like stats
            count = stats[0].flatten()[-1]
            mean = stats[0, :-1] / count
            var = stats[1, :-1] / count - mean * mean
        else:
            # New style: Npz file
            count = stats["count"]
            sum_v = stats["sum"]
            sum_square_v = stats["sum_square"]
            mean = sum_v / count
            var = sum_square_v / count - mean * mean
        std = np.sqrt(np.maximum(var, eps))

        if isinstance(mean, np.ndarray):
            mean = torch.from_numpy(mean)
        else:
            mean = torch.tensor(mean).float()
        if isinstance(std, np.ndarray):
            std = torch.from_numpy(std)
        else:
            std = torch.tensor(std).float()

        self.register_buffer("mean", mean)
        self.register_buffer("std", std)

    def extra_repr(self):
        """
            Generate a string representation of the GlobalMVN instance.

        This method provides a concise representation of the instance, including
        the path to the statistics file and the normalization settings for means
        and variances. It is primarily used for debugging and logging purposes.

        Returns:
            str: A string representation of the GlobalMVN instance, showing
            the stats file, norm_means, and norm_vars attributes.

        Examples:
            >>> mvn = GlobalMVN("path/to/stats.npy", norm_means=True, norm_vars=False)
            >>> print(mvn.extra_repr())
            stats_file=path/to/stats.npy, norm_means=True, norm_vars=False
        """
        return (
            f"stats_file={self.stats_file}, "
            f"norm_means={self.norm_means}, norm_vars={self.norm_vars}"
        )

    def forward(
        self, x: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Forward function

        This method applies global mean and variance normalization to the input tensor
        `x`. It adjusts the input based on the computed mean and standard deviation,
        which are derived from the statistics loaded from the specified `stats_file`.

        Args:
            x (torch.Tensor): Input tensor of shape (B, L, ...), where B is the batch
                size and L is the sequence length.
            ilens (torch.Tensor, optional): Tensor of shape (B,) that contains the
                actual lengths of each sequence in the batch. If not provided, it
                defaults to the length of `x`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
                - The normalized tensor of the same shape as `x`.
                - The tensor `ilens` that represents the lengths of each sequence.

        Examples:
            >>> mvn = GlobalMVN("path/to/stats.npy")
            >>> x = torch.randn(2, 5, 10)  # Example input
            >>> ilens = torch.tensor([5, 3])  # Example lengths
            >>> normalized_x, normalized_ilens = mvn.forward(x, ilens)
        """
        if ilens is None:
            ilens = x.new_full([x.size(0)], x.size(1))
        norm_means = self.norm_means
        norm_vars = self.norm_vars
        self.mean = self.mean.to(x.device, x.dtype)
        self.std = self.std.to(x.device, x.dtype)
        mask = make_pad_mask(ilens, x, 1)

        # feat: (B, T, D)
        if norm_means:
            if x.requires_grad:
                x = x - self.mean
            else:
                x -= self.mean
        if x.requires_grad:
            x = x.masked_fill(mask, 0.0)
        else:
            x.masked_fill_(mask, 0.0)

        if norm_vars:
            x /= self.std

        return x, ilens

    def inverse(
        self, x: torch.Tensor, ilens: torch.Tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
            Apply global mean and variance normalization.

        This class implements global mean and variance normalization, which can be
        used to standardize the input data based on precomputed statistics. The
        normalization can be applied to both the means and variances, with an
        optional small epsilon value to prevent division by zero.

        TODO(kamo): Make this class portable somehow.

        Attributes:
            stats_file (Path): Path to the .npy file containing the statistics.
            norm_means (bool): Whether to apply mean normalization.
            norm_vars (bool): Whether to apply variance normalization.
            eps (float): Small value to prevent division by zero.
            mean (torch.Tensor): Mean values for normalization.
            std (torch.Tensor): Standard deviation values for normalization.

        Args:
            stats_file: npy file containing the statistics for normalization.
            norm_means: Apply mean normalization (default: True).
            norm_vars: Apply variance normalization (default: True).
            eps: Small value to prevent division by zero (default: 1.0e-20).

        Examples:
            >>> global_mvn = GlobalMVN(stats_file='stats.npy')
            >>> normalized_tensor, lengths = global_mvn.forward(input_tensor)
            >>> original_tensor, lengths = global_mvn.inverse(normalized_tensor)

        Raises:
            ValueError: If the provided statistics file cannot be loaded or is
            invalid.
        """
        if ilens is None:
            ilens = x.new_full([x.size(0)], x.size(1))
        norm_means = self.norm_means
        norm_vars = self.norm_vars
        self.mean = self.mean.to(x.device, x.dtype)
        self.std = self.std.to(x.device, x.dtype)
        mask = make_pad_mask(ilens, x, 1)

        if x.requires_grad:
            x = x.masked_fill(mask, 0.0)
        else:
            x.masked_fill_(mask, 0.0)

        if norm_vars:
            x *= self.std

        # feat: (B, T, D)
        if norm_means:
            x += self.mean
            x.masked_fill_(make_pad_mask(ilens, x, 1), 0.0)
        return x, ilens
