"""Helpers for complex-valued tensors.

Everything in espnet2 that handles a spectrum uses PyTorch's own complex
tensors (``torch.complex64`` / ``torch.complex128``). This module holds the
few operations that need care - building a complex tensor from parts whose
dtype PyTorch's constructor refuses, mixed real/complex products, a batched
trace - and the acceptance of ``torch_complex.ComplexTensor``, the class
espnet used before PyTorch had complex support: when that package happens to
be installed, its tensors are recognised by :func:`is_complex` and converted
by :func:`to_complex`, so a caller that still builds one gets a native tensor
back. Nothing here requires the package.
"""

from typing import Sequence, Tuple

import torch

try:  # accepted on input only; never produced
    from torch_complex.tensor import ComplexTensor
except ImportError:  # pragma: no cover - the package is optional
    ComplexTensor = None

EPS = torch.finfo(torch.double).eps


def _is_legacy(c) -> bool:
    return ComplexTensor is not None and isinstance(c, ComplexTensor)


def complex_tensor(real: torch.Tensor, imag: torch.Tensor) -> torch.Tensor:
    """Build a complex tensor from real and imaginary parts.

    ``torch.complex`` accepts only float16, float32 and float64 parts, and
    complex32 supports few operators; parts in any other dtype, bfloat16 in
    particular, are widened to float32 first.
    """
    if real.dtype not in (torch.float32, torch.float64):
        real = real.float()
    if imag.dtype != real.dtype:
        imag = imag.to(real.dtype)
    return torch.complex(real, imag)


def new_complex_like(
    ref: torch.Tensor,
    real_imag: Tuple[torch.Tensor, torch.Tensor],
) -> torch.Tensor:
    """Build a complex tensor from parts; ``ref`` is kept for API compatibility."""
    return complex_tensor(*real_imag)


def is_torch_complex_tensor(c) -> bool:
    return isinstance(c, torch.Tensor) and torch.is_complex(c)


def is_complex(c) -> bool:
    return is_torch_complex_tensor(c) or _is_legacy(c)


def to_complex(c) -> torch.Tensor:
    """Return ``c`` as a native complex tensor.

    Accepts a complex tensor (returned as is), a legacy ComplexTensor, or a
    real tensor whose last dimension holds (real, imag).
    """
    if _is_legacy(c):
        return complex_tensor(c.real, c.imag)
    if torch.is_complex(c):
        return c
    return torch.view_as_complex(c)


def to_double(c: torch.Tensor) -> torch.Tensor:
    if is_complex(c):
        return to_complex(c).to(dtype=torch.complex128)
    return c.double()


def to_float(c: torch.Tensor) -> torch.Tensor:
    if is_complex(c):
        return to_complex(c).to(dtype=torch.complex64)
    return c.float()


def _native(seq: Sequence[torch.Tensor]):
    return [to_complex(x) if _is_legacy(x) else x for x in seq]


def cat(seq: Sequence[torch.Tensor], *args, **kwargs) -> torch.Tensor:
    if not isinstance(seq, (list, tuple)):
        raise TypeError(
            "cat(): argument 'tensors' (position 1) must be tuple of Tensors, "
            "not Tensor"
        )
    return torch.cat(_native(seq), *args, **kwargs)


def stack(seq: Sequence[torch.Tensor], *args, **kwargs) -> torch.Tensor:
    if not isinstance(seq, (list, tuple)):
        raise TypeError(
            "stack(): argument 'tensors' (position 1) must be tuple of Tensors, "
            "not Tensor"
        )
    return torch.stack(_native(seq), *args, **kwargs)


def complex_norm(c: torch.Tensor, dim=-1, keepdim=False) -> torch.Tensor:
    if not is_complex(c):
        raise TypeError("Input is not a complex tensor.")
    return torch.norm(to_complex(c), dim=dim, keepdim=keepdim)


def _mixed(op, a: torch.Tensor, b: torch.Tensor, *args) -> torch.Tensor:
    """Apply a bilinear ``op`` when exactly one operand may be real.

    torch.einsum / torch.matmul promote a real operand to complex themselves
    since PyTorch 1.9, but doing the two real products keeps the result
    identical to what the code produced before and costs nothing.
    """
    if torch.is_complex(a) and not torch.is_complex(b):
        return torch.complex(op(*args, a.real, b), op(*args, a.imag, b))
    if torch.is_complex(b) and not torch.is_complex(a):
        return torch.complex(op(*args, a, b.real), op(*args, a, b.imag))
    return op(*args, a, b)


def einsum(equation: str, *operands) -> torch.Tensor:
    if len(operands) == 1 and isinstance(operands[0], (tuple, list)):
        operands = tuple(operands[0])
    operands = tuple(_native(operands))
    if len(operands) == 2:
        return _mixed(torch.einsum, operands[0], operands[1], equation)
    if len(operands) < 2 or len({op.dtype for op in operands}) != 1:
        raise ValueError("0 or More than 2 operands are not supported.")
    return torch.einsum(equation, *operands)


def matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = _native((a, b))
    return _mixed(torch.matmul, a, b)


def inverse(c: torch.Tensor) -> torch.Tensor:
    return torch.linalg.inv(to_complex(c) if _is_legacy(c) else c)


def trace(a: torch.Tensor) -> torch.Tensor:
    """Batched trace over the last two dimensions: (..., N, N) -> (...)."""
    if _is_legacy(a):
        a = to_complex(a)
    return torch.diagonal(a, dim1=-2, dim2=-1).sum(-1)


def reverse(a: torch.Tensor, dim=0) -> torch.Tensor:
    if _is_legacy(a):
        a = to_complex(a)
    return torch.flip(a, dims=(dim,))


def solve(b: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
    """Solve the linear equation ax = b."""
    a, b = _native((a, b))
    if torch.is_complex(a) != torch.is_complex(b):
        return matmul(inverse(a), b)
    return torch.linalg.solve(a, b)
