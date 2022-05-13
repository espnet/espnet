from distutils.version import LooseVersion

import numpy as np
import pytest
import torch
import torch_complex.functional as FC
from torch_complex.tensor import ComplexTensor

from espnet2.enh.layers.complex_utils import cat
from espnet2.enh.layers.complex_utils import complex_norm
from espnet2.enh.layers.complex_utils import einsum
from espnet2.enh.layers.complex_utils import inverse
from espnet2.enh.layers.complex_utils import matmul
from espnet2.enh.layers.complex_utils import solve
from espnet2.enh.layers.complex_utils import stack
from espnet2.enh.layers.complex_utils import trace


is_torch_1_9_plus = LooseVersion(torch.__version__) >= LooseVersion("1.9.0")
# invertible matrix
mat_np = np.array(
    [
        [
            [-0.211 + 1.8293j, -0.1138 + 0.0754j, -1.3574 - 0.6358j],
            [-1.1041 - 1.0455j, -0.8856 - 0.7828j, 1.6058 + 0.8616j],
            [0.3877 - 1.3823j, 1.2027 - 0.4265j, 0.4436 - 0.0173j],
        ],
        [
            [0.5322 - 0.2629j, 1.774 - 0.9664j, -0.1956 + 0.8791j],
            [-0.156 - 0.1044j, 0.2576 + 1.2311j, 0.0493 - 2.5577j],
            [0.4465 - 1.1056j, 0.4398 + 1.4871j, -0.34 + 1.095j],
        ],
    ],
    dtype=np.complex64,
)


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_cat(dim):
    if is_torch_1_9_plus:
        wrappers = [ComplexTensor, torch.complex]
        modules = [FC, torch]
    else:
        wrappers = [ComplexTensor]
        modules = [FC]

    for complex_wrapper, complex_module in zip(wrappers, modules):
        mat1 = complex_wrapper(torch.rand(2, 3, 4), torch.rand(2, 3, 4))
        mat2 = complex_wrapper(torch.rand(2, 3, 4), torch.rand(2, 3, 4))
        ret = cat([mat1, mat2], dim=dim)
        ret2 = complex_module.cat([mat1, mat2], dim=dim)
        assert complex_module.allclose(ret, ret2)


@pytest.mark.parametrize("dim", [None, 0, 1, 2])
@pytest.mark.skipif(not is_torch_1_9_plus, reason="Require torch 1.9.0+")
def test_complex_norm(dim):
    mat = ComplexTensor(torch.rand(2, 3, 4), torch.rand(2, 3, 4))
    mat_th = torch.complex(mat.real, mat.imag)
    norm = complex_norm(mat, dim=dim, keepdim=True)
    norm_th = complex_norm(mat_th, dim=dim, keepdim=True)
    assert torch.allclose(norm, norm_th)
    if dim is not None:
        assert norm.ndim == mat.ndim and mat.numel() == norm.numel() * mat.size(dim)


@pytest.mark.parametrize("real_vec", [True, False])
def test_einsum(real_vec):
    if is_torch_1_9_plus:
        wrappers = [ComplexTensor, torch.complex]
        modules = [FC, torch]
    else:
        wrappers = [ComplexTensor]
        modules = [FC]

    for complex_wrapper, complex_module in zip(wrappers, modules):
        mat = complex_wrapper(torch.rand(2, 3, 3), torch.rand(2, 3, 3))
        if real_vec:
            vec = torch.rand(2, 3, 1)
            vec2 = complex_wrapper(vec, torch.zeros_like(vec))
        else:
            vec = complex_wrapper(torch.rand(2, 3, 1), torch.rand(2, 3, 1))
            vec2 = vec
        ret = einsum("bec,bcf->bef", mat, vec)
        ret2 = complex_module.einsum("bec,bcf->bef", mat, vec2)
        assert complex_module.allclose(ret, ret2)


def test_inverse():
    if is_torch_1_9_plus:
        wrappers = [ComplexTensor, torch.complex]
        modules = [FC, torch]
    else:
        wrappers = [ComplexTensor]
        modules = [FC]

    eye = torch.eye(3).expand(2, 3, 3)
    for complex_wrapper, complex_module in zip(wrappers, modules):
        mat = complex_wrapper(
            torch.from_numpy(mat_np.real), torch.from_numpy(mat_np.imag)
        )
        eye_complex = complex_wrapper(eye, torch.zeros_like(eye))
        assert complex_module.allclose(mat @ inverse(mat), eye_complex, atol=1e-6)


@pytest.mark.parametrize("real_vec", [True, False])
def test_matmul(real_vec):
    if is_torch_1_9_plus:
        wrappers = [ComplexTensor, torch.complex]
        modules = [FC, torch]
    else:
        wrappers = [ComplexTensor]
        modules = [FC]

    for complex_wrapper, complex_module in zip(wrappers, modules):
        mat = complex_wrapper(torch.rand(2, 3, 3), torch.rand(2, 3, 3))
        if real_vec:
            vec = torch.rand(2, 3, 1)
            vec2 = complex_wrapper(vec, torch.zeros_like(vec))
        else:
            vec = complex_wrapper(torch.rand(2, 3, 1), torch.rand(2, 3, 1))
            vec2 = vec
        ret = matmul(mat, vec)
        ret2 = complex_module.matmul(mat, vec2)
        assert complex_module.allclose(ret, ret2)


def test_trace():
    if is_torch_1_9_plus:
        wrappers = [ComplexTensor, torch.complex]
        modules = [FC, torch]
    else:
        wrappers = [ComplexTensor]
        modules = [FC]

    for complex_wrapper, complex_module in zip(wrappers, modules):
        mat = complex_wrapper(torch.rand(2, 3, 3), torch.rand(2, 3, 3))
        tr = trace(mat)
        tr2 = sum([mat[..., i, i] for i in range(mat.size(-1))])
        assert complex_module.allclose(tr, tr2)


@pytest.mark.parametrize("real_vec", [True, False])
def test_solve(real_vec):
    if is_torch_1_9_plus:
        wrappers = [ComplexTensor, torch.complex]
        modules = [FC, torch]
    else:
        wrappers = [ComplexTensor]
        modules = [FC]

    for complex_wrapper, complex_module in zip(wrappers, modules):
        mat = complex_wrapper(
            torch.from_numpy(mat_np.real), torch.from_numpy(mat_np.imag)
        )
        if not real_vec or complex_wrapper is ComplexTensor:
            vec = complex_wrapper(torch.rand(2, 3, 1), torch.rand(2, 3, 1))
            vec2 = vec
        else:
            vec = torch.rand(2, 3, 1)
            vec2 = complex_wrapper(vec, torch.zeros_like(vec))
        ret = solve(vec, mat)
        if isinstance(vec2, ComplexTensor):
            ret2 = FC.solve(vec2, mat, return_LU=False)
        else:
            return torch.linalg.solve(mat, vec2)
        assert complex_module.allclose(ret, ret2)


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_stack(dim):
    if is_torch_1_9_plus:
        wrappers = [ComplexTensor, torch.complex]
        modules = [FC, torch]
    else:
        wrappers = [ComplexTensor]
        modules = [FC]

    for complex_wrapper, complex_module in zip(wrappers, modules):
        print(complex_wrapper, complex_module)
        mat1 = complex_wrapper(torch.rand(2, 3, 4), torch.rand(2, 3, 4))
        mat2 = complex_wrapper(torch.rand(2, 3, 4), torch.rand(2, 3, 4))
        ret = stack([mat1, mat2], dim=dim)
        ret2 = complex_module.stack([mat1, mat2], dim=dim)
        assert complex_module.allclose(ret, ret2)


def test_complex_impl_consistency():
    if not is_torch_1_9_plus:
        return
    mat_th = torch.complex(torch.from_numpy(mat_np.real), torch.from_numpy(mat_np.imag))
    mat_ct = ComplexTensor(torch.from_numpy(mat_np.real), torch.from_numpy(mat_np.imag))
    bs = mat_th.shape[0]
    rank = mat_th.shape[-1]
    vec_th = torch.complex(torch.rand(bs, rank), torch.rand(bs, rank)).type_as(mat_th)
    vec_ct = ComplexTensor(vec_th.real, vec_th.imag)

    for result_th, result_ct in (
        (abs(mat_th), abs(mat_ct)),
        (inverse(mat_th), inverse(mat_ct)),
        (matmul(mat_th, vec_th.unsqueeze(-1)), matmul(mat_ct, vec_ct.unsqueeze(-1))),
        (solve(vec_th.unsqueeze(-1), mat_th), solve(vec_ct.unsqueeze(-1), mat_ct)),
        (
            einsum("bec,bc->be", mat_th, vec_th),
            einsum("bec,bc->be", mat_ct, vec_ct),
        ),
    ):
        np.testing.assert_allclose(result_th.numpy(), result_ct.numpy(), atol=1e-6)
