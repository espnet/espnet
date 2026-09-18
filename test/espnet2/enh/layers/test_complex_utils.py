import numpy as np
import pytest
import torch

from espnet2.enh.layers.complex_utils import (
    cat,
    complex_norm,
    einsum,
    inverse,
    matmul,
    solve,
    stack,
    trace,
)

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
    wrappers = [torch.complex]
    modules = [torch]

    for complex_wrapper, complex_module in zip(wrappers, modules):
        mat1 = complex_wrapper(torch.rand(2, 3, 4), torch.rand(2, 3, 4))
        mat2 = complex_wrapper(torch.rand(2, 3, 4), torch.rand(2, 3, 4))
        ret = cat([mat1, mat2], dim=dim)
        ret2 = complex_module.cat([mat1, mat2], dim=dim)
        assert complex_module.allclose(ret, ret2)


@pytest.mark.parametrize("dim", [None, 0, 1, 2])
def test_complex_norm(dim):
    mat = torch.complex(torch.rand(2, 3, 4), torch.rand(2, 3, 4))
    mat_th = torch.complex(mat.real, mat.imag)
    norm = complex_norm(mat, dim=dim, keepdim=True)
    norm_th = complex_norm(mat_th, dim=dim, keepdim=True)
    assert torch.allclose(norm, norm_th)
    if dim is not None:
        assert norm.ndim == mat.ndim and mat.numel() == norm.numel() * mat.size(dim)


@pytest.mark.parametrize("real_vec", [True, False])
def test_einsum(real_vec):
    wrappers = [torch.complex]
    modules = [torch]

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
    wrappers = [torch.complex]
    modules = [torch]

    eye = torch.eye(3).expand(2, 3, 3)
    for complex_wrapper, complex_module in zip(wrappers, modules):
        mat = complex_wrapper(
            torch.from_numpy(mat_np.real), torch.from_numpy(mat_np.imag)
        )
        eye_complex = complex_wrapper(eye, torch.zeros_like(eye))
        assert complex_module.allclose(mat @ inverse(mat), eye_complex, atol=1e-6)


@pytest.mark.parametrize("real_vec", [True, False])
def test_matmul(real_vec):
    wrappers = [torch.complex]
    modules = [torch]

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
    wrappers = [torch.complex]
    modules = [torch]

    for complex_wrapper, complex_module in zip(wrappers, modules):
        mat = complex_wrapper(torch.rand(2, 3, 3), torch.rand(2, 3, 3))
        tr = trace(mat)
        tr2 = sum([mat[..., i, i] for i in range(mat.size(-1))])
        assert complex_module.allclose(tr, tr2)


@pytest.mark.parametrize("real_vec", [True, False])
def test_solve(real_vec):
    wrappers = [torch.complex]
    modules = [torch]

    for complex_wrapper, complex_module in zip(wrappers, modules):
        mat = complex_wrapper(
            torch.from_numpy(mat_np.real), torch.from_numpy(mat_np.imag)
        )
        if not real_vec:
            vec = complex_wrapper(torch.rand(2, 3, 1), torch.rand(2, 3, 1))
            vec2 = vec
        else:
            vec = torch.rand(2, 3, 1)
            vec2 = complex_wrapper(vec, torch.zeros_like(vec))
        ret = solve(vec, mat)
        ret2 = torch.linalg.solve(mat, vec2)
        assert complex_module.allclose(ret, ret2)


@pytest.mark.parametrize("dim", [0, 1, 2])
def test_stack(dim):
    wrappers = [torch.complex]
    modules = [torch]

    for complex_wrapper, complex_module in zip(wrappers, modules):
        print(complex_wrapper, complex_module)
        mat1 = complex_wrapper(torch.rand(2, 3, 4), torch.rand(2, 3, 4))
        mat2 = complex_wrapper(torch.rand(2, 3, 4), torch.rand(2, 3, 4))
        ret = stack([mat1, mat2], dim=dim)
        ret2 = complex_module.stack([mat1, mat2], dim=dim)
        assert complex_module.allclose(ret, ret2)


def test_against_numpy():
    # numpy is the independent oracle: inverse, matmul, solve and einsum on
    # the fixed matrix above must agree with it
    torch.random.manual_seed(0)
    mat = torch.from_numpy(mat_np)
    vec_np = (np.random.rand(2, 3, 1) + 1j * np.random.rand(2, 3, 1)).astype(
        np.complex64
    )
    vec = torch.from_numpy(vec_np)
    np.testing.assert_allclose(inverse(mat).numpy(), np.linalg.inv(mat_np), atol=1e-5)
    np.testing.assert_allclose(matmul(mat, vec).numpy(), mat_np @ vec_np, atol=1e-5)
    np.testing.assert_allclose(
        solve(vec, mat).numpy(), np.linalg.solve(mat_np, vec_np), atol=1e-4
    )
    np.testing.assert_allclose(
        einsum("bec,bcf->bef", mat, vec).numpy(),
        np.einsum("bec,bcf->bef", mat_np, vec_np),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        trace(mat).numpy(), np.trace(mat_np, axis1=-2, axis2=-1), atol=1e-6
    )


def test_legacy_complextensor_is_accepted_when_installed():
    torch_complex = pytest.importorskip("torch_complex")
    from espnet2.enh.layers.complex_utils import is_complex, to_complex

    mat = torch.from_numpy(mat_np)
    legacy = torch_complex.tensor.ComplexTensor(mat.real, mat.imag)
    assert is_complex(legacy)
    assert torch.allclose(to_complex(legacy), mat)
    assert torch.allclose(matmul(legacy, mat), mat @ mat)
    assert torch.allclose(trace(legacy), trace(mat))


def test_legacy_complextensor_accepted_at_public_boundaries():
    torch_complex = pytest.importorskip("torch_complex")
    from espnet2.enh.decoder.stft_decoder import STFTDecoder
    from espnet2.enh.encoder.stft_encoder import STFTEncoder
    from espnet2.enh.layers.beamformer import signal_framing
    from espnet2.enh.layers.wpe import wpe_one_iteration

    torch.random.manual_seed(0)
    x = torch.randn(2, 400)
    enc, dec = STFTEncoder(n_fft=64, hop_length=16), STFTDecoder(
        n_fft=64, hop_length=16
    )
    spec, flens = enc(x, torch.tensor([400, 400]))
    legacy = torch_complex.tensor.ComplexTensor(spec.real, spec.imag)
    # the decoder converts a legacy spectrum and gives the same waveform
    wav_native, _ = dec(spec, flens)
    wav_legacy, _ = dec(legacy, flens)
    assert torch.allclose(wav_native, wav_legacy)
    # framing and WPE take the legacy form too, and return native tensors
    framed = signal_framing(legacy, 3, 1, 1)
    assert torch.is_complex(framed)
    Y = spec.permute(2, 0, 1)  # (F, C=batch, T)
    Yl = torch_complex.tensor.ComplexTensor(Y.real, Y.imag)
    power = (Y.real**2 + Y.imag**2).mean(-2)  # (F, T)
    out_native = wpe_one_iteration(Y, power, taps=2, delay=1)
    out_legacy = wpe_one_iteration(Yl, power, taps=2, delay=1)
    assert torch.is_complex(out_legacy) and torch.allclose(out_native, out_legacy)


def test_use_builtin_complex_false_is_deprecated_not_honoured():
    from espnet2.enh.encoder.stft_encoder import STFTEncoder

    with pytest.warns(DeprecationWarning, match="use_builtin_complex"):
        enc = STFTEncoder(n_fft=64, hop_length=16, use_builtin_complex=False)
    spec, _ = enc(torch.randn(1, 200), torch.tensor([200]))
    assert torch.is_complex(spec)
