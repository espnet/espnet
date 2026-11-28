"""Signal operations shared by DNF, MixIT and SNOAR training.

Ported from the SNOAR reference implementation (bDNF, branch
``fix/resumable-legs-and-fixed-mixtures`` at 2b1e186):
``bdnf/losses/sisdr.py``, ``bdnf/probes/projection.py``,
``bdnf/bootstrap/remix.py``, ``bdnf/models/consistency.py`` and the remix-level
helpers of ``run_whamr_compare.py``. All functions act on the last axis, which
is time, and keep the reference's conventions:

* SI-SDR with ``eps = 1e-8`` and no mean removal.
* The DNF projection (eq. 14 of arXiv:2606.02327) uses one coefficient per
  utterance, guarded by ``tau = tau_rel * mean_batch(||n_hat||^2)``.
* ``align_scale`` pins the free SI-SDR gain of a pseudo-label to the gain of
  the observation it was estimated from.
"""

from __future__ import annotations

import math

import torch

EPS = 1e-8


def _dot(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return torch.sum(a * b, dim=-1, keepdim=True)


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
def compute_si_sdr(
    est: torch.Tensor, ref: torch.Tensor, eps: float = EPS
) -> torch.Tensor:
    """Return the scale-invariant SDR in dB, one value per row.

    Args:
        est: Estimate, shape ``(..., T)``.
        ref: Reference, shape ``(..., T)``.
        eps: Numerical floor.

    Returns:
        Tensor of shape ``(...)``.

    Examples:
        >>> x = torch.randn(2, 100)
        >>> bool((compute_si_sdr(x, x) > 60).all())
        True
    """
    alpha = _dot(est, ref) / (_dot(ref, ref) + eps)
    proj = alpha * ref
    err = est - proj
    ratio = _dot(proj, proj) / (_dot(err, err) + eps)
    return 10.0 * torch.log10(ratio.squeeze(-1) + eps)


def compute_sisdr_loss(
    est: torch.Tensor, ref: torch.Tensor, scale_w: float = 0.0
) -> torch.Tensor:
    """Return the batch mean of negative SI-SDR plus an optional scale anchor.

    ``scale_w > 0`` adds ``scale_w * (alpha - 1)^2`` with
    ``alpha = <est, ref> / ||ref||^2``. The reference trains every arm with
    ``scale_w = 0``, i.e. plain negative SI-SDR.
    """
    loss = -compute_si_sdr(est, ref)
    if scale_w > 0.0:
        alpha = _dot(est, ref) / (_dot(ref, ref) + EPS)
        loss = loss + scale_w * ((alpha - 1.0) ** 2).squeeze(-1)
    return loss.mean()


def compute_mixit_loss(
    o0: torch.Tensor,
    o1: torch.Tensor,
    o2: torch.Tensor,
    m1: torch.Tensor,
    m2: torch.Tensor,
) -> torch.Tensor:
    """Return the constrained MixIT speech-enhancement loss.

    Wisdom et al. 2020 (arXiv:2006.12701), enhancement setup: the model sees
    ``m1 + m2`` (``m1`` noisy speech, ``m2`` noise only) and emits one speech
    slot ``o0`` and two noise slots. ``m1`` is rebuilt from ``o0`` plus one
    noise slot and ``m2`` from the other, so speech is never assigned to the
    noise-only mixture. The best of the two assignments is taken per example.
    """
    score_a = compute_si_sdr(o0 + o1, m1) + compute_si_sdr(o2, m2)
    score_b = compute_si_sdr(o0 + o2, m1) + compute_si_sdr(o1, m2)
    return (-torch.maximum(score_a, score_b)).mean()


# --------------------------------------------------------------------------- #
# DNF projection and pseudo-label operators
# --------------------------------------------------------------------------- #
def project(
    s_hat: torch.Tensor, n_hat: torch.Tensor, tau_rel: float = 1e-3
) -> torch.Tensor:
    """Return the DNF projection ``s_hat - c * n_hat``.

    ``c = <n_hat, s_hat> / (||n_hat||^2 + tau)`` with
    ``tau = tau_rel * mean_batch(||n_hat||^2)``, i.e. one global coefficient
    per utterance. The coefficient is invariant to a rescaling of ``n_hat``,
    so the free SI-SDR gain of the noise head does not matter.
    """
    n_energy = _dot(n_hat, n_hat)
    tau = tau_rel * n_energy.mean().detach()
    c = _dot(n_hat, s_hat) / (n_energy + tau + EPS)
    return s_hat - c * n_hat


def decontaminate(n_hat: torch.Tensor, s_ref: torch.Tensor) -> torch.Tensor:
    """Remove from ``n_hat`` its component along ``s_ref``."""
    coef = _dot(n_hat, s_ref) / (_dot(s_ref, s_ref) + EPS)
    return n_hat - coef * s_ref


def align_scale(x: torch.Tensor, ref: torch.Tensor) -> torch.Tensor:
    """Scale ``x`` by ``<x, ref> / ||x||^2``, its least-squares gain onto ``ref``.

    The observation ``ref`` has speech gain 1 by construction, so this pins
    the free SI-SDR gain of a teacher output before it is reused as a target.
    """
    scale = _dot(x, ref) / (_dot(x, x) + EPS)
    return scale * x


def add_observation(
    s_hat: torch.Tensor, x_obs: torch.Tensor, gamma: float
) -> torch.Tensor:
    """Return the observation-added target ``(1 - gamma) s_hat + gamma x_obs``."""
    return (1.0 - gamma) * s_hat + gamma * x_obs


def normalize_gain(x: torch.Tensor, target_energy: torch.Tensor) -> torch.Tensor:
    """Scale each row of ``x`` to the energy ``target_energy``."""
    return x * torch.sqrt(target_energy / (_dot(x, x) + EPS))


def scale_to_snr(
    s_ref: torch.Tensor, noise: torch.Tensor, snr_db: torch.Tensor
) -> torch.Tensor:
    """Scale ``noise`` per row to the SNR ``snr_db`` (dB) against ``s_ref``.

    After scaling, ``10 log10(||s_ref||^2 / ||noise||^2) == snr_db``.
    """
    es = _dot(s_ref, s_ref)
    en = _dot(noise, noise).clamp_min(EPS)
    return noise * torch.sqrt(es / (en * 10 ** (snr_db.unsqueeze(-1) / 10)))


def pair_by_roll(n: int, device, min_shift: int = 1) -> torch.Tensor:
    """Return a circular-shift permutation that pairs every row with another."""
    shift = max(1, min_shift)
    return (torch.arange(n, device=device) + shift) % n


def anneal_gamma(gen: int, n_gen: int, gamma: float, gamma_end: float) -> float:
    """Return the observation-adding weight for generation ``gen`` (1-based).

    Cosine anneal from ``gamma`` at generation 1 to ``gamma_end`` at
    generation ``n_gen``; generations after ``n_gen`` stay at ``gamma_end``.
    """
    gen = min(max(gen, 1), max(n_gen, 1))
    cos = 0.5 * (1.0 + math.cos(math.pi * (gen - 1) / max(1, n_gen - 1)))
    return float(gamma_end + (gamma - gamma_end) * cos)


# --------------------------------------------------------------------------- #
# Mixture consistency (Wisdom et al. 2019, arXiv:1811.08521)
# --------------------------------------------------------------------------- #
def apply_mixture_consistency(
    sources: torch.Tensor,
    mixture: torch.Tensor,
    mode: str = "energy",
    n_fft: int = 512,
    hop: int = 128,
) -> torch.Tensor:
    """Project ``sources`` (B, J, T) so that they sum to ``mixture`` (B, T).

    ``uniform`` splits the residual equally (eq. 4). ``energy`` splits it per
    time-frequency bin by source power (eq. 5), which needs an STFT round
    trip. ``global`` uses one power weight per source and utterance.
    ``none`` returns ``sources`` unchanged.
    """
    if mode == "none":
        return sources
    B, J, T = sources.shape
    if mode == "uniform":
        resid = mixture - sources.sum(dim=1)
        return sources + resid.unsqueeze(1) / J
    if mode == "global":
        resid = mixture - sources.sum(dim=1)
        e = (sources**2).sum(dim=-1, keepdim=True)
        w = e / (e.sum(dim=1, keepdim=True) + EPS)
        return sources + w * resid.unsqueeze(1)
    if mode != "energy":
        raise ValueError(f"unknown mixture-consistency mode {mode!r}")
    win = torch.hann_window(n_fft, device=sources.device, dtype=torch.float32)
    S = torch.stft(
        sources.reshape(B * J, T).float(), n_fft, hop, window=win, return_complex=True
    )
    S = S.reshape(B, J, *S.shape[-2:])
    Y = torch.stft(mixture.float(), n_fft, hop, window=win, return_complex=True)
    resid = Y - S.sum(dim=1)
    p = S.abs() ** 2
    w = p / (p.sum(dim=1, keepdim=True) + EPS)
    Sc = S + w * resid.unsqueeze(1)
    out = torch.istft(
        Sc.reshape(B * J, *Sc.shape[-2:]), n_fft, hop, window=win, length=T
    )
    return out.reshape(B, J, T).to(sources.dtype)
