"""Flow-related transformation.

This code is derived from https://github.com/bayesiains/nflows.

"""

import numpy as np
import torch
from torch.nn import functional as F

DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3


# TODO(kan-bayashi): Documentation and type hint
def piecewise_rational_quadratic_transform(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails=None,
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    """
        Applies a piecewise rational quadratic transformation to the input data.

    This transformation is useful for creating flexible, piecewise-defined
    functions that can model complex distributions. It utilizes the concept of
    rational quadratic splines to perform the transformation, allowing for both
    forward and inverse operations.

    Attributes:
        DEFAULT_MIN_BIN_WIDTH (float): Default minimum width for bins.
        DEFAULT_MIN_BIN_HEIGHT (float): Default minimum height for bins.
        DEFAULT_MIN_DERIVATIVE (float): Default minimum derivative value.

    Args:
        inputs (torch.Tensor): The input tensor to transform.
        unnormalized_widths (torch.Tensor): Unnormalized widths for the spline bins.
        unnormalized_heights (torch.Tensor): Unnormalized heights for the spline bins.
        unnormalized_derivatives (torch.Tensor): Unnormalized derivatives for the
            spline bins.
        inverse (bool, optional): If True, applies the inverse transformation.
            Defaults to False.
        tails (str or None, optional): Defines the behavior of the tails of the
            spline. If None, a rational quadratic spline is used. If 'linear',
            linear tails are applied. Defaults to None.
        tail_bound (float, optional): The boundary for the tails. Defaults to 1.0.
        min_bin_width (float, optional): Minimum allowable width for bins.
            Defaults to DEFAULT_MIN_BIN_WIDTH.
        min_bin_height (float, optional): Minimum allowable height for bins.
            Defaults to DEFAULT_MIN_BIN_HEIGHT.
        min_derivative (float, optional): Minimum allowable derivative value.
            Defaults to DEFAULT_MIN_DERIVATIVE.

    Returns:
        tuple: A tuple containing:
            - outputs (torch.Tensor): The transformed output tensor.
            - logabsdet (torch.Tensor): The log absolute determinant of the
              transformation.

    Raises:
        ValueError: If the input tensor is outside the domain defined by the
            spline or if the minimum bin dimensions are not feasible.

    Examples:
        >>> inputs = torch.tensor([0.1, 0.5, 0.9])
        >>> unnormalized_widths = torch.tensor([[0.1, 0.2], [0.2, 0.3]])
        >>> unnormalized_heights = torch.tensor([[0.5, 0.6], [0.6, 0.7]])
        >>> unnormalized_derivatives = torch.tensor([[0.01, 0.02], [0.02, 0.03]])
        >>> outputs, logabsdet = piecewise_rational_quadratic_transform(
        ...     inputs, unnormalized_widths, unnormalized_heights,
        ...     unnormalized_derivatives, inverse=False
        ... )
    """
    if tails is None:
        spline_fn = rational_quadratic_spline
        spline_kwargs = {}
    else:
        spline_fn = unconstrained_rational_quadratic_spline
        spline_kwargs = {"tails": tails, "tail_bound": tail_bound}

    outputs, logabsdet = spline_fn(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        **spline_kwargs
    )
    return outputs, logabsdet


# TODO(kan-bayashi): Documentation and type hint
def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    """
        Unconstrained Rational Quadratic Spline Transformation.

    This function implements an unconstrained rational quadratic spline for
    transformation, which can be used in various applications, such as
    normalizing flows. The transformation allows for flexible mapping of
    inputs to outputs while maintaining differentiability.

    Attributes:
        inputs (torch.Tensor): The input tensor to be transformed.
        unnormalized_widths (torch.Tensor): The unnormalized widths for the
            spline bins.
        unnormalized_heights (torch.Tensor): The unnormalized heights for the
            spline bins.
        unnormalized_derivatives (torch.Tensor): The unnormalized derivatives
            at the spline knots.
        inverse (bool): If True, performs the inverse transformation.
        tails (str): Specifies the behavior of the tails; default is "linear".
        tail_bound (float): The bounds for the tails of the spline.
        min_bin_width (float): The minimum allowed width of the spline bins.
        min_bin_height (float): The minimum allowed height of the spline bins.
        min_derivative (float): The minimum allowed derivative for the spline.

    Args:
        inputs (torch.Tensor): The input values to be transformed.
        unnormalized_widths (torch.Tensor): Unnormalized widths of the spline.
        unnormalized_heights (torch.Tensor): Unnormalized heights of the spline.
        unnormalized_derivatives (torch.Tensor): Unnormalized derivatives at the
            spline knots.
        inverse (bool, optional): If True, apply the inverse transformation.
            Defaults to False.
        tails (str, optional): Specifies the behavior of the tails. Defaults
            to "linear".
        tail_bound (float, optional): Bound for the tails. Defaults to 1.0.
        min_bin_width (float, optional): Minimum bin width. Defaults to
            DEFAULT_MIN_BIN_WIDTH.
        min_bin_height (float, optional): Minimum bin height. Defaults to
            DEFAULT_MIN_BIN_HEIGHT.
        min_derivative (float, optional): Minimum derivative. Defaults to
            DEFAULT_MIN_DERIVATIVE.

    Returns:
        tuple: A tuple containing:
            - outputs (torch.Tensor): The transformed output values.
            - logabsdet (torch.Tensor): The log absolute determinant of the
              Jacobian of the transformation.

    Raises:
        RuntimeError: If the specified tails are not implemented.

    Examples:
        >>> inputs = torch.tensor([0.5, 0.6])
        >>> widths = torch.tensor([[0.1, 0.2], [0.1, 0.2]])
        >>> heights = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
        >>> derivatives = torch.tensor([[0.1, 0.1], [0.1, 0.1]])
        >>> outputs, logabsdet = unconstrained_rational_quadratic_spline(
        ...     inputs, widths, heights, derivatives
        ... )

    Note:
        The function assumes that the input tensors are properly shaped
        and compatible with each other.

    Todo:
        - Enhance documentation and add type hints.
    """
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    (
        outputs[inside_interval_mask],
        logabsdet[inside_interval_mask],
    ) = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    return outputs, logabsdet


# TODO(kan-bayashi): Documentation and type hint
def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    """
    Applies a rational quadratic spline transformation to the inputs.

    This function computes the rational quadratic spline transformation for a
    given set of inputs based on specified widths, heights, and derivatives.
    The transformation can be performed in both forward and inverse directions
    based on the `inverse` parameter.

    Args:
        inputs (torch.Tensor): The input tensor to be transformed.
        unnormalized_widths (torch.Tensor): The unnormalized widths of the bins.
        unnormalized_heights (torch.Tensor): The unnormalized heights of the bins.
        unnormalized_derivatives (torch.Tensor): The unnormalized derivatives
            at the bin edges.
        inverse (bool, optional): If True, performs the inverse transformation.
            Defaults to False.
        left (float, optional): The left boundary of the input range.
            Defaults to 0.0.
        right (float, optional): The right boundary of the input range.
            Defaults to 1.0.
        bottom (float, optional): The bottom boundary of the output range.
            Defaults to 0.0.
        top (float, optional): The top boundary of the output range.
            Defaults to 1.0.
        min_bin_width (float, optional): The minimum allowed width for the bins.
            Defaults to 1e-3.
        min_bin_height (float, optional): The minimum allowed height for the bins.
            Defaults to 1e-3.
        min_derivative (float, optional): The minimum allowed derivative value.
            Defaults to 1e-3.

    Returns:
        tuple: A tuple containing:
            - outputs (torch.Tensor): The transformed output tensor.
            - logabsdet (torch.Tensor): The log absolute determinant of the
              transformation.

    Raises:
        ValueError: If the input is not within the specified domain or if the
            minimal bin width or height is too large for the number of bins.

    Examples:
        >>> inputs = torch.tensor([0.1, 0.5, 0.9])
        >>> widths = torch.tensor([[0.1, 0.2], [0.1, 0.2], [0.1, 0.2]])
        >>> heights = torch.tensor([[0.3, 0.4], [0.3, 0.4], [0.3, 0.4]])
        >>> derivatives = torch.tensor([[0.1, 0.1], [0.1, 0.1], [0.1, 0.1]])
        >>> outputs, logabsdet = rational_quadratic_spline(inputs, widths,
        ... heights, derivatives)
        >>> print(outputs)
        >>> print(logabsdet)

    Note:
        The implementation relies on the properties of rational quadratic
        splines, which can be beneficial in various applications such as
        normalizing flows in generative models.
    """
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input to a transform is not within its domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = _searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = _searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (
            input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta
        )
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta)
            * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet


def _searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1
