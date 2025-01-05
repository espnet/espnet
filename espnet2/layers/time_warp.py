"""Time warp module."""

import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list

DEFAULT_TIME_WARP_MODE = "bicubic"


def time_warp(x: torch.Tensor, window: int = 80, mode: str = DEFAULT_TIME_WARP_MODE):
    """
        Time warp module.

    This module provides functions and classes for performing time warping on
    tensor data using PyTorch's interpolation capabilities.

    The `time_warp` function applies a time warping effect to the input tensor
    `x` based on the specified window size and interpolation mode. The `TimeWarp`
    class encapsulates this functionality, allowing for easy integration into
    neural network models.

    Attributes:
        DEFAULT_TIME_WARP_MODE (str): The default interpolation mode, set to
        "bicubic".

    Args:
        x (torch.Tensor): Input tensor of shape (Batch, Time, Freq).
        window (int): Time warp parameter that defines the range of the warp.
        Default is 80.
        mode (str): Interpolation mode to be used for warping. Default is
        DEFAULT_TIME_WARP_MODE.

    Returns:
        torch.Tensor: The time-warped tensor of the same shape as the input.

    Raises:
        ValueError: If the input tensor does not have at least 3 dimensions.

    Examples:
        >>> import torch
        >>> x = torch.rand(2, 100, 64)  # (Batch, Time, Freq)
        >>> warped_x = time_warp(x, window=50, mode='linear')

        >>> time_warp_layer = TimeWarp(window=50, mode='linear')
        >>> output, lengths = time_warp_layer(x)

    Note:
        Bicubic interpolation supports tensors with 4 or more dimensions.

    Todo:
        - Implement a batch processing strategy for varying input lengths.
    """

    # bicubic supports 4D or more dimension tensor
    org_size = x.size()
    if x.dim() == 3:
        # x: (Batch, Time, Freq) -> (Batch, 1, Time, Freq)
        x = x[:, None]

    t = x.shape[2]
    if t - window <= window:
        return x.view(*org_size)

    center = torch.randint(window, t - window, (1,))[0]
    warped = torch.randint(center - window, center + window, (1,))[0] + 1

    # left: (Batch, Channel, warped, Freq)
    # right: (Batch, Channel, time - warped, Freq)
    left = torch.nn.functional.interpolate(
        x[:, :, :center], (warped, x.shape[3]), mode=mode, align_corners=False
    )
    right = torch.nn.functional.interpolate(
        x[:, :, center:], (t - warped, x.shape[3]), mode=mode, align_corners=False
    )

    if x.requires_grad:
        x = torch.cat([left, right], dim=-2)
    else:
        x[:, :, :warped] = left
        x[:, :, warped:] = right

    return x.view(*org_size)


class TimeWarp(torch.nn.Module):
    """
        Time warp module for temporal interpolation of audio features using PyTorch.

    This module provides functionality for time warping audio feature tensors
    using various interpolation modes. It includes both a standalone function
    for time warping and a PyTorch module that can be integrated into a
    neural network pipeline.

    The `time_warp` function performs time warping on a given tensor, while
    the `TimeWarp` class allows for more flexible integration and usage within
    a neural network model.

    Args:
        x (torch.Tensor): Input tensor of shape (Batch, Time, Freq).
        window (int, optional): Time warp parameter that controls the extent of
            warping. Default is 80.
        mode (str, optional): Interpolation mode. Default is "bicubic".

    Returns:
        torch.Tensor: The time-warped tensor of the same shape as input.

    Examples:
        # Using the standalone function
        import torch
        x = torch.randn(2, 100, 64)  # Example input tensor
        warped_x = time_warp(x, window=80, mode='linear')

        # Using the TimeWarp module in a model
        time_warp_layer = TimeWarp(window=80, mode='linear')
        output, lengths = time_warp_layer(x)

    Note:
        The `time_warp` function can handle tensors of shape (Batch, Time, Freq)
        or (Batch, 1, Time, Freq) when using bicubic interpolation. It will
        reshape the input tensor accordingly.

    Raises:
        ValueError: If the input tensor's time dimension is not sufficient for
            the specified window size.

    Todo:
        - Implement batch processing for variable-length sequences in the
            TimeWarp class's forward method.
    """

    def __init__(self, window: int = 80, mode: str = DEFAULT_TIME_WARP_MODE):
        super().__init__()
        self.window = window
        self.mode = mode

    def extra_repr(self):
        """
                Returns a string representation of the TimeWarp module's parameters.

        This method is used to provide additional information about the instance
        of the TimeWarp class when printed. It includes the window size and
        interpolation mode that are set during initialization.

        Attributes:
            window (int): The time warp parameter.
            mode (str): The interpolation mode used for warping.

        Returns:
            str: A formatted string containing the window and mode values.

        Examples:
            >>> tw = TimeWarp(window=100, mode='linear')
            >>> print(tw.extra_repr())
            window=100, mode=linear
        """
        return f"window={self.window}, mode={self.mode}"

    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor = None):
        """
                Forward function for the TimeWarp module.

        This method applies time warping to the input tensor using the specified
        window size and interpolation mode. It supports variable-length input
        sequences, ensuring that the same warping is applied to each sample when
        lengths are uniform.

        Args:
            x: A tensor of shape (Batch, Time, Freq) representing the input data.
            x_lengths: A tensor of shape (Batch,) containing the lengths of each
                sequence in the batch. If None, the same warping is applied to each
                sample.

        Returns:
            A tuple containing:
                - A tensor with the warped output of shape (Batch, Time, Freq).
                - A tensor with the lengths of each sequence in the batch.

        Examples:
            # Example usage with uniform lengths
            import torch
            model = TimeWarp(window=80, mode='bicubic')
            input_tensor = torch.randn(10, 100, 40)  # (Batch, Time, Freq)
            output_tensor, lengths = model(input_tensor)

            # Example usage with variable lengths
            input_tensor_var_len = torch.randn(5, 120, 40)  # (Batch, Time, Freq)
            lengths = torch.tensor([100, 90, 80, 70, 60])  # Variable lengths
            output_tensor_var_len, lengths_var_len = model(input_tensor_var_len, lengths)
        """

        if x_lengths is None or all(le == x_lengths[0] for le in x_lengths):
            # Note that applying same warping for each sample
            y = time_warp(x, window=self.window, mode=self.mode)
        else:
            # FIXME(kamo): I have no idea to batchify Timewarp
            ys = []
            for i in range(x.size(0)):
                _y = time_warp(
                    x[i][None, : x_lengths[i]],
                    window=self.window,
                    mode=self.mode,
                )[0]
                ys.append(_y)
            y = pad_list(ys, 0.0)

        return y, x_lengths
