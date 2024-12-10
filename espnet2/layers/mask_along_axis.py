import math
from typing import Sequence, Union

import torch
from typeguard import typechecked


def mask_along_axis(
    spec: torch.Tensor,
    spec_lengths: torch.Tensor,
    mask_width_range: Sequence[int] = (0, 30),
    dim: int = 1,
    num_mask: int = 2,
    replace_with_zero: bool = True,
):
    """
        Apply a mask along a specified axis of a tensor, randomly masking out
    portions of the input tensor based on the specified parameters.

    This function generates a random mask for the specified dimension of the
    input tensor and applies it, either replacing the masked values with
    zeros or the mean of the tensor, depending on the `replace_with_zero`
    parameter.

    Args:
        spec (torch.Tensor): Input tensor of shape (Batch, Length, Freq).
        spec_lengths (torch.Tensor): Lengths of the input sequences, not used
            in this implementation. Shape: (Length).
        mask_width_range (Sequence[int], optional): A tuple specifying the
            minimum and maximum width of the mask. The width is chosen randomly
            from this range. Defaults to (0, 30).
        dim (int, optional): The dimension along which to apply the mask.
            Defaults to 1 (Length).
        num_mask (int, optional): The number of masks to apply. Defaults to 2.
        replace_with_zero (bool, optional): If True, masked values will be
            replaced with zeros. If False, masked values will be replaced with
            the mean of the input tensor. Defaults to True.

    Returns:
        tuple: A tuple containing:
            - torch.Tensor: The masked tensor of shape (Batch, Length, Freq).
            - torch.Tensor: The original spec_lengths tensor.

    Examples:
        >>> import torch
        >>> spec = torch.rand(4, 10, 20)  # Batch of 4, Length 10, Freq 20
        >>> spec_lengths = torch.tensor([10, 10, 10, 10])
        >>> masked_spec, lengths = mask_along_axis(spec, spec_lengths)
        >>> masked_spec.shape
        torch.Size([4, 10, 20])

    Raises:
        ValueError: If `dim` is not 1 or 2.
        TypeError: If `mask_width_range` is not a tuple of two integers.
    """

    org_size = spec.size()
    if spec.dim() == 4:
        # spec: (Batch, Channel, Length, Freq) -> (Batch * Channel, Length, Freq)
        spec = spec.view(-1, spec.size(2), spec.size(3))

    B = spec.shape[0]
    # D = Length or Freq
    D = spec.shape[dim]
    # mask_length: (B, num_mask, 1)
    mask_length = torch.randint(
        mask_width_range[0],
        mask_width_range[1],
        (B, num_mask),
        device=spec.device,
    ).unsqueeze(2)

    # mask_pos: (B, num_mask, 1)
    mask_pos = torch.randint(
        0, max(1, D - mask_length.max()), (B, num_mask), device=spec.device
    ).unsqueeze(2)

    # aran: (1, 1, D)
    aran = torch.arange(D, device=spec.device)[None, None, :]
    # mask: (Batch, num_mask, D)
    mask = (mask_pos <= aran) * (aran < (mask_pos + mask_length))
    # Multiply masks: (Batch, num_mask, D) -> (Batch, D)
    mask = mask.any(dim=1)
    if dim == 1:
        # mask: (Batch, Length, 1)
        mask = mask.unsqueeze(2)
    elif dim == 2:
        # mask: (Batch, 1, Freq)
        mask = mask.unsqueeze(1)

    if replace_with_zero:
        value = 0.0
    else:
        value = spec.mean()

    if spec.requires_grad:
        spec = spec.masked_fill(mask, value)
    else:
        spec = spec.masked_fill_(mask, value)
    spec = spec.view(*org_size)
    return spec, spec_lengths


class MaskAlongAxis(torch.nn.Module):
    """
        Mask input tensor along a specified axis with random masking.

    This class provides functionality to apply random masking to input tensors,
    allowing for more robust training of models by simulating missing data.

    Attributes:
        mask_width_range (Union[int, Sequence[int]]): The range of widths for the
            masks. Can be a single integer or a tuple defining the minimum and
            maximum width.
        num_mask (int): The number of masks to apply to the input tensor.
        dim (Union[int, str]): The dimension along which to apply the mask. Can be
            specified as an integer (1 for time, 2 for frequency) or as a string.
        replace_with_zero (bool): Whether to replace the masked values with zeros
            or with the mean of the tensor.

    Args:
        mask_width_range (Union[int, Sequence[int]]): The range of widths for the
            masks.
        num_mask (int): The number of masks to apply.
        dim (Union[int, str]): The dimension along which to mask ('time' or
            'freq').
        replace_with_zero (bool): Flag to determine the value used to replace
            masked elements.

    Examples:
        >>> mask_layer = MaskAlongAxis(mask_width_range=(0, 20), num_mask=3)
        >>> masked_spec, lengths = mask_layer(spec_tensor, spec_lengths)

    Raises:
        TypeError: If mask_width_range is not a tuple of two integers.
        ValueError: If dim is not an integer or one of the specified strings.

    Note:
        The masking is performed in-place for tensors that do not require gradients.
    """

    @typechecked
    def __init__(
        self,
        mask_width_range: Union[int, Sequence[int]] = (0, 30),
        num_mask: int = 2,
        dim: Union[int, str] = "time",
        replace_with_zero: bool = True,
    ):
        if isinstance(mask_width_range, int):
            mask_width_range = (0, mask_width_range)
        if len(mask_width_range) != 2:
            raise TypeError(
                f"mask_width_range must be a tuple of int and int values: "
                f"{mask_width_range}",
            )

        assert mask_width_range[1] > mask_width_range[0]
        if isinstance(dim, str):
            if dim == "time":
                dim = 1
            elif dim == "freq":
                dim = 2
            else:
                raise ValueError("dim must be int, 'time' or 'freq'")
        if dim == 1:
            self.mask_axis = "time"
        elif dim == 2:
            self.mask_axis = "freq"
        else:
            self.mask_axis = "unknown"

        super().__init__()
        self.mask_width_range = mask_width_range
        self.num_mask = num_mask
        self.dim = dim
        self.replace_with_zero = replace_with_zero

    def extra_repr(self):
        """
                Returns a string representation of the MaskAlongAxis module's attributes.

        The string representation includes the mask width range, the number of masks, and
        the masking axis used in the masking process.

        Attributes:
            mask_width_range (Union[int, Sequence[int]]): The range of mask widths.
            num_mask (int): The number of masks to apply.
            mask_axis (str): The axis along which the masking is applied, either
                "time" or "freq".

        Returns:
            str: A formatted string representation of the module's parameters.

        Examples:
            >>> mask_layer = MaskAlongAxis(mask_width_range=(5, 10), num_mask=3)
            >>> print(mask_layer.extra_repr())
            mask_width_range=(5, 10), num_mask=3, axis=time
        """
        return (
            f"mask_width_range={self.mask_width_range}, "
            f"num_mask={self.num_mask}, axis={self.mask_axis}"
        )

    def forward(self, spec: torch.Tensor, spec_lengths: torch.Tensor = None):
        """Forward function.

        Args:
            spec: (Batch, Length, Freq)
        """

        return mask_along_axis(
            spec,
            spec_lengths,
            mask_width_range=self.mask_width_range,
            dim=self.dim,
            num_mask=self.num_mask,
            replace_with_zero=self.replace_with_zero,
        )


class MaskAlongAxisVariableMaxWidth(torch.nn.Module):
    """
    Mask input spec along a specified axis with variable maximum width.

    This module applies a masking operation to the input tensor along a
    specified axis. The maximum width of the mask is determined by the
    ratio of the input sequence length. The mask is applied randomly
    within the range defined by the mask width ratio.

    Formula:
        max_width = max_width_ratio * seq_len

    Attributes:
        mask_width_ratio_range: A tuple of floats defining the minimum and
            maximum ratios for the mask width relative to the sequence
            length. The default is (0.0, 0.05).
        num_mask: An integer defining the number of masks to apply.
            The default is 2.
        dim: An integer or string defining the axis along which to apply
            the mask. Accepts 1 for time or 2 for frequency. The default
            is "time".
        replace_with_zero: A boolean indicating whether to replace masked
            values with zero or the mean of the tensor. The default is
            True.

    Args:
        mask_width_ratio_range: (Union[float, Sequence[float]]):
            Range of mask width ratios. Default is (0.0, 0.05).
        num_mask: (int): Number of masks to apply. Default is 2.
        dim: (Union[int, str]): Axis to apply mask. Can be 'time' (1) or
            'freq' (2). Default is 'time'.
        replace_with_zero: (bool): If True, replace masked values with
            zero. Default is True.

    Raises:
        TypeError: If mask_width_ratio_range is not a tuple of floats.
        ValueError: If dim is not an int, 'time', or 'freq'.

    Examples:
        >>> mask_layer = MaskAlongAxisVariableMaxWidth(num_mask=3, dim=2)
        >>> spec = torch.randn(5, 100, 80)  # Batch of 5, Length 100, Freq 80
        >>> masked_spec, lengths = mask_layer(spec)

    Note:
        The input tensor must have at least 3 dimensions.
    """

    @typechecked
    def __init__(
        self,
        mask_width_ratio_range: Union[float, Sequence[float]] = (0.0, 0.05),
        num_mask: int = 2,
        dim: Union[int, str] = "time",
        replace_with_zero: bool = True,
    ):
        if isinstance(mask_width_ratio_range, float):
            mask_width_ratio_range = (0.0, mask_width_ratio_range)
        if len(mask_width_ratio_range) != 2:
            raise TypeError(
                f"mask_width_ratio_range must be a tuple of float and float values: "
                f"{mask_width_ratio_range}",
            )

        assert mask_width_ratio_range[1] > mask_width_ratio_range[0]
        if isinstance(dim, str):
            if dim == "time":
                dim = 1
            elif dim == "freq":
                dim = 2
            else:
                raise ValueError("dim must be int, 'time' or 'freq'")
        if dim == 1:
            self.mask_axis = "time"
        elif dim == 2:
            self.mask_axis = "freq"
        else:
            self.mask_axis = "unknown"

        super().__init__()
        self.mask_width_ratio_range = mask_width_ratio_range
        self.num_mask = num_mask
        self.dim = dim
        self.replace_with_zero = replace_with_zero

    def extra_repr(self):
        """
            Returns a string representation of the MaskAlongAxisVariableMaxWidth
        instance's parameters.

        This method provides a concise summary of the key attributes of the
        instance, which include the range of mask widths, the number of masks,
        and the axis along which the masking occurs.

        Attributes:
            mask_width_ratio_range: The range of the mask width ratio used to
                determine the maximum width of the mask relative to the input
                sequence length.
            num_mask: The number of masks to apply to the input tensor.
            mask_axis: The axis along which the masking will be applied,
                represented as either "time" or "freq".

        Returns:
            A string summarizing the instance's parameters.

        Examples:
            >>> mask_layer = MaskAlongAxisVariableMaxWidth(mask_width_ratio_range=(0.1, 0.2),
            ...                                              num_mask=3,
            ...                                              dim='time',
            ...                                              replace_with_zero=False)
            >>> print(mask_layer.extra_repr())
            mask_width_ratio_range=(0.1, 0.2), num_mask=3, axis=time
        """
        return (
            f"mask_width_ratio_range={self.mask_width_ratio_range}, "
            f"num_mask={self.num_mask}, axis={self.mask_axis}"
        )

    def forward(self, spec: torch.Tensor, spec_lengths: torch.Tensor = None):
        """
        Forward function.

        Applies masking to the input tensor along the specified axis with
        variable maximum width determined by the mask width ratio range.

        Args:
            spec: A tensor of shape (Batch, Length, Freq) representing the input
                data to be masked.
            spec_lengths: Optional tensor representing the lengths of the
                sequences in the batch. Default is None.

        Returns:
            A tuple containing:
                - The masked tensor of the same shape as `spec`.
                - The original `spec_lengths` tensor.

        Examples:
            >>> mask = MaskAlongAxisVariableMaxWidth(mask_width_ratio_range=(0.0, 0.1))
            >>> input_spec = torch.randn(4, 100, 80)  # Batch of 4, Length 100, Freq 80
            >>> masked_spec, lengths = mask(input_spec)

        Note:
            The masking is applied only if the maximum mask width is greater
            than the minimum mask width.

        Raises:
            ValueError: If `dim` is not an integer or not one of "time" or
            "freq".
        """

        max_seq_len = spec.shape[self.dim]
        min_mask_width = math.floor(max_seq_len * self.mask_width_ratio_range[0])
        min_mask_width = max([0, min_mask_width])
        max_mask_width = math.floor(max_seq_len * self.mask_width_ratio_range[1])
        max_mask_width = min([max_seq_len, max_mask_width])

        if max_mask_width > min_mask_width:
            return mask_along_axis(
                spec,
                spec_lengths,
                mask_width_range=(min_mask_width, max_mask_width),
                dim=self.dim,
                num_mask=self.num_mask,
                replace_with_zero=self.replace_with_zero,
            )
        return spec, spec_lengths
