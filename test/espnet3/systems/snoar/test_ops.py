import pytest
import torch

from espnet3.systems.snoar.src.ops import (
    add_observation,
    align_scale,
    anneal_gamma,
    apply_mixture_consistency,
    compute_mixit_loss,
    compute_si_sdr,
    decontaminate,
    project,
    scale_to_snr,
)


def test_projection_removes_noise_component():
    s, n = torch.randn(2, 1000), torch.randn(2, 1000)
    out = project(s + 0.4 * n, 3.0 * n, tau_rel=0.0)  # noise head gain is free
    assert torch.allclose(
        out,
        s - (s * n).sum(-1, keepdim=True) / (n * n).sum(-1, keepdim=True) * n,
        atol=1e-4,
    )


def test_decontaminate_is_orthogonal_and_align_pins_gain():
    s, n = torch.randn(1, 500), torch.randn(1, 500)
    nd = decontaminate(n + 0.5 * s, s)
    assert abs(float((nd * s).sum())) < 1e-3
    x = s + 0.1 * n
    assert torch.allclose(align_scale(4.3 * s, x), align_scale(s, x), atol=1e-5)


def test_si_sdr_and_mixit_loss():
    s = torch.randn(3, 800)
    assert bool((compute_si_sdr(2.0 * s, s) > 60).all())
    n_a, n_b = torch.randn(3, 800), torch.randn(3, 800)
    perfect = compute_mixit_loss(s, n_a, n_b, s + n_a, n_b)
    swapped = compute_mixit_loss(s, n_b, n_a, s + n_a, n_b)  # assignment B wins
    assert float(perfect) < -60 and float(swapped) < -60


def test_scale_to_snr_and_observation_adding():
    s, n = torch.randn(2, 400), torch.randn(2, 400)
    out = scale_to_snr(s, n, torch.tensor([0.0, 10.0]))
    snr = 10 * torch.log10((s**2).sum(-1) / (out**2).sum(-1))
    assert torch.allclose(snr, torch.tensor([0.0, 10.0]), atol=1e-3)
    assert torch.allclose(add_observation(s, s + n, 0.2), s + 0.2 * n, atol=1e-6)


def test_gamma_anneal_endpoints():
    assert anneal_gamma(1, 16, 0.2, 0.05) == pytest.approx(0.2)
    assert anneal_gamma(16, 16, 0.2, 0.05) == pytest.approx(0.05)
    assert anneal_gamma(40, 16, 0.2, 0.05) == pytest.approx(0.05)
    assert anneal_gamma(1, 1, 0.2, 0.05) == pytest.approx(0.2)


@pytest.mark.parametrize("mode", ["uniform", "global", "energy"])
def test_mixture_consistency_sums_to_mixture(mode):
    x = torch.randn(2, 2048)
    srcs = torch.randn(2, 3, 2048)
    out = apply_mixture_consistency(srcs, x, mode=mode, n_fft=256, hop=64)
    assert torch.allclose(out.sum(1), x, atol=1e-3)


def test_mixture_consistency_rejects_unknown_mode():
    with pytest.raises(ValueError, match="unknown mixture-consistency mode"):
        apply_mixture_consistency(torch.randn(1, 2, 100), torch.randn(1, 100), "bad")
