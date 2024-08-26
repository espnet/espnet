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
    """Compute Cauchy kernel.

    v: (B, N)
    z: (L)
    w: (B, N)
    symmetric: whether to assume that v and w contain complex conjugate pairs, of the
    form [v_half, v_half.conj()] and [w_half, w_half.conj()]
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
    """Wrap the cuda method to deal with shapes."""
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
    @staticmethod
    def forward(ctx, v, z, w):
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
        v, z, w = ctx.saved_tensors
        dv, dw = cauchy_mult_bwd(v, z, w, dout)
        return dv, None, dw


class CauchyMultiplySymmetric(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, z, w):
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
        v, z, w = ctx.saved_tensors
        dv, dw = cauchy_mult_sym_bwd(v, z, w, dout)
        return dv, None, dw
