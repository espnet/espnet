# This code is derived from https://github.com/HazyResearch/state-spaces

import torch
from cauchy_mult import (
    cauchy_mult_bwd,
    cauchy_mult_fwd,
    cauchy_mult_sym_bwd,
    cauchy_mult_sym_fwd,
)
from einops import rearrange


def cauchy_mult_torch(
    v: torch.Tensor, z: torch.Tensor, w: torch.Tensor, symmetric=True
) -> torch.Tensor:
    """
    Compute the Cauchy kernel using PyTorch operations.

    This function calculates the Cauchy kernel for given input tensors. It supports
    both symmetric and non-symmetric computations.

    Args:
        v (torch.Tensor): Input tensor of shape (B, N), where B is the batch size
            and N is the number of elements.
        z (torch.Tensor): Input tensor of shape (L,), where L is the length.
        w (torch.Tensor): Input tensor of shape (B, N), where B is the batch size
            and N is the number of elements.
        symmetric (bool, optional): If True, assumes v and w contain complex conjugate
            pairs of the form [v_half, v_half.conj()] and [w_half, w_half.conj()].
            Defaults to True.

    Returns:
        torch.Tensor: The computed Cauchy kernel.

    Raises:
        AssertionError: If symmetric is True and N is not even.

    Examples:
        >>> v = torch.randn(2, 4)
        >>> z = torch.randn(3)
        >>> w = torch.randn(2, 4)
        >>> result = cauchy_mult_torch(v, z, w)
        >>> print(result.shape)
        torch.Size([2, 3])

    Note:
        When symmetric is True, the function only uses the first half of v and w,
        assuming they contain complex conjugate pairs.
    """
    if not symmetric:
        return (
            rearrange(v, "b n -> b 1 n")
            / (rearrange(z, "l -> l 1") - rearrange(w, "b n -> b 1 n"))
        ).sum(dim=-1)
    else:
        N = v.shape[-1]
        assert N % 2 == 0
        vv = rearrange(v[:, : N // 2], "b n -> b 1 n")
        zz = rearrange(z, "l -> l 1")
        ww = rearrange(w[:, : N // 2], "b n -> b 1 n")
        return 2 * (
            (zz * vv.real - vv.real * ww.real - vv.imag * ww.imag)
            / (zz * zz - 2 * zz * ww.real + ww.abs().square())
        ).sum(dim=-1)


def cauchy_mult_keops(v, z, w):
    """
    Compute the Cauchy kernel using KeOps LazyTensors.

    This function calculates the Cauchy kernel for given input tensors using
    KeOps LazyTensors for efficient GPU computation.

    Args:
        v (torch.Tensor): Input tensor of shape (b, N), where b is the batch size
            and N is the number of elements.
        z (torch.Tensor): Input tensor of shape (L,), where L is the length.
        w (torch.Tensor): Input tensor of shape (b, N), where b is the batch size
            and N is the number of elements.

    Returns:
        torch.Tensor: The computed Cauchy kernel of shape (b, L).

    Note:
        This function requires the PyKeOps library to be installed and
        configured properly for GPU acceleration.

    Examples:
        >>> v = torch.randn(2, 1000)
        >>> z = torch.randn(500)
        >>> w = torch.randn(2, 1000)
        >>> result = cauchy_mult_keops(v, z, w)
        >>> print(result.shape)
        torch.Size([2, 500])
    """
    from pykeops.torch import LazyTensor

    v_l = LazyTensor(rearrange(v, "b N -> b 1 N 1"))
    z_l = LazyTensor(rearrange(z, "L -> 1 L 1 1"))
    w_l = LazyTensor(rearrange(w, "b N -> b 1 N 1"))
    sub = z_l - w_l  # (b N L 1), for some reason it doesn't display the last dimension
    div = v_l / sub
    s = div.sum(dim=2, backend="GPU")
    return s.squeeze(-1)


def _cauchy_mult(v, z, w, symmetric=True):
    if not symmetric:
        return CauchyMultiply.apply(v, z, w)
    else:
        return CauchyMultiplySymmetric.apply(v, z, w)


def cauchy_mult(v, z, w, symmetric=True):
    """
    Wrapper function for Cauchy multiplication that handles tensor shapes.

    This function wraps the CUDA-based Cauchy multiplication method, ensuring proper
    tensor shapes and broadcasting before computation.

    Args:
        v (torch.Tensor): Input tensor of shape (..., N), where N is the number of elements.
        z (torch.Tensor): Input tensor of shape (L,), where L is the length.
        w (torch.Tensor): Input tensor of shape (..., N), where N is the number of elements.
        symmetric (bool, optional): If True, uses symmetric Cauchy multiplication.
            Defaults to True.

    Returns:
        torch.Tensor: The result of Cauchy multiplication with shape (..., L).

    Raises:
        AssertionError: If z is not a 1-dimensional tensor after squeezing.
        AssertionError: If the last dimensions of v and w do not match.

    Note:
        This function broadcasts v and w tensors, making it flexible for various
        input shapes.

    Examples:
        >>> v = torch.randn(2, 3, 1000)
        >>> z = torch.randn(500)
        >>> w = torch.randn(2, 3, 1000)
        >>> result = cauchy_mult(v, z, w)
        >>> print(result.shape)
        torch.Size([2, 3, 500])
    """
    v, w = torch.broadcast_tensors(v, w)
    shape = v.shape
    # z_shape = z.shape
    z = z.squeeze()
    assert len(z.shape) == 1

    v = v.contiguous()
    w = w.contiguous()
    z = z.contiguous()

    N = v.size(-1)
    assert w.size(-1) == N
    y = _cauchy_mult(v.view(-1, N), z, w.view(-1, N), symmetric=symmetric)
    y = y.view(*shape[:-1], z.size(-1))
    # y = z.new_zeros(*shape[:-1], z.size(-1))
    return y


class CauchyMultiply(torch.autograd.Function):
    """
    Custom autograd function for Cauchy multiplication.

    This class implements a custom autograd function for Cauchy multiplication,
    providing both forward and backward passes. It utilizes CUDA operations
    for efficient computation on GPU.

    Note:
        This class is designed to be used with PyTorch's autograd system and
        should not be instantiated directly. Instead, use it through the
        `_cauchy_mult` function.

    Attributes:
        Inherits attributes from torch.autograd.Function.

    Raises:
        NotImplementedError: If input tensors are not CUDA tensors, or if the
            input dimensions are not supported.
    """

    @staticmethod
    def forward(ctx, v, z, w):
        """
            Perform the forward pass of Cauchy multiplication.

        This method computes the Cauchy multiplication using CUDA operations.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Context object to save
                information for backward computation.
            v (torch.Tensor): Input tensor of shape (batch, N).
            z (torch.Tensor): Input tensor of shape (L,).
            w (torch.Tensor): Input tensor of shape (batch, N).

        Returns:
            torch.Tensor: Result of Cauchy multiplication.

        Raises:
            NotImplementedError: If N is not in the supported values list.
            NotImplementedError: If L is not a multiple of 32.
            NotImplementedError: If input tensors are not CUDA tensors.

        Note:
            Currently only supports N values of 64 (2^6) and L values that are
            multiples of 32.
        """
        batch, N = v.shape
        # supported_N_values = [1 << log_n for log_n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        supported_N_values = [1 << log_n for log_n in [6]]
        L = z.shape[-1]
        if N not in supported_N_values:
            raise NotImplementedError(f"Only support N values in {supported_N_values}")
        if L % 32 != 0:
            raise NotImplementedError("Only support L values that are multiples of 32")
        if not v.is_cuda and z.is_cuda and w.is_cuda:
            raise NotImplementedError("Only support CUDA tensors")
        ctx.save_for_backward(v, z, w)
        return cauchy_mult_fwd(v, z, w)

    @staticmethod
    def backward(ctx, dout):
        """
            Perform the backward pass of Cauchy multiplication.

        This method computes the gradients with respect to inputs v and w.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Context object containing
                saved tensors from the forward pass.
            dout (torch.Tensor): Gradient of the loss with respect to the output
                of the forward pass.

        Returns:
            tuple: A tuple containing:
                - dv (torch.Tensor): Gradient with respect to input v.
                - None: Placeholder for gradient with respect to z (not computed).
                - dw (torch.Tensor): Gradient with respect to input w.

        Note:
            The gradient with respect to z is not computed and returned as None.
        """
        v, z, w = ctx.saved_tensors
        dv, dw = cauchy_mult_bwd(v, z, w, dout)
        return dv, None, dw


class CauchyMultiplySymmetric(torch.autograd.Function):
    """
    Custom autograd function for symmetric Cauchy multiplication.

    This class implements a custom autograd function for symmetric Cauchy
    multiplication, providing both forward and backward passes. It utilizes
    CUDA operations for efficient computation on GPU, assuming symmetry in
    the input tensors.

    Note:
        This class is designed to be used with PyTorch's autograd system and
        should not be instantiated directly. Instead, use it through the
        `_cauchy_mult` function with the symmetric option set to True.

    Attributes:
        Inherits attributes from torch.autograd.Function.

    Raises:
        NotImplementedError: If input tensors are not CUDA tensors, or if the
            input dimensions are not supported.
    """

    @staticmethod
    def forward(ctx, v, z, w):
        """
            Perform the forward pass of symmetric Cauchy multiplication.

        This method computes the symmetric Cauchy multiplication using CUDA operations.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Context object to save
                information for backward computation.
            v (torch.Tensor): Input tensor of shape (batch, N).
            z (torch.Tensor): Input tensor of shape (L,).
            w (torch.Tensor): Input tensor of shape (batch, N).

        Returns:
            torch.Tensor: Result of symmetric Cauchy multiplication.

        Raises:
            NotImplementedError: If N is not in the supported values list.
            NotImplementedError: If L exceeds the maximum supported value.
            NotImplementedError: If input tensors are not CUDA tensors.

        Note:
            Supports N values that are powers of 2 from 2 to 1024.
            The maximum supported L value is 32 * 1024 * 64 * 1024.
        """
        batch, N = v.shape
        supported_N_values = [1 << log_n for log_n in [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        L = z.shape[-1]
        if N not in supported_N_values:
            raise NotImplementedError(f"Only support N values in {supported_N_values}")
        max_L_value = 32 * 1024 * 64 * 1024
        if L > max_L_value:
            raise NotImplementedError("Only support L values <= {max_L_value}")
        if not v.is_cuda and z.is_cuda and w.is_cuda:
            raise NotImplementedError("Only support CUDA tensors")
        ctx.save_for_backward(v, z, w)
        return cauchy_mult_sym_fwd(v, z, w)

    @staticmethod
    def backward(ctx, dout):
        """
            Perform the backward pass of symmetric Cauchy multiplication.

        This method computes the gradients with respect to inputs v and w for the
        symmetric case.

        Args:
            ctx (torch.autograd.function.FunctionCtx): Context object containing
                saved tensors from the forward pass.
            dout (torch.Tensor): Gradient of the loss with respect to the output
                of the forward pass.

        Returns:
            tuple: A tuple containing:
                - dv (torch.Tensor): Gradient with respect to input v.
                - None: Placeholder for gradient with respect to z (not computed).
                - dw (torch.Tensor): Gradient with respect to input w.

        Note:
            The gradient with respect to z is not computed and returned as None.
            This method assumes symmetry in the input tensors and uses specialized
            CUDA operations for symmetric Cauchy multiplication.
        """
        v, z, w = ctx.saved_tensors
        dv, dw = cauchy_mult_sym_bwd(v, z, w, dout)
        return dv, None, dw
