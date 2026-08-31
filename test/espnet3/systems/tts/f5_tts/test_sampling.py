"""Sampling path: the fixed-step solvers, ``CFM.sample`` and ``F5TTS.inference``.

The training tests only exercise ``forward``. Everything reached during
generation - the ODE loop, the classifier-free-guidance branch, the mel
extraction inside ``CFM.sample`` - is covered here instead.
"""

import pytest
import torch

from espnet3.systems.tts.f5_tts.f5tts import F5TTS
from espnet3.systems.tts.f5_tts.solvers import odeint

MODEL_CONF = dict(
    hidden_size=32,
    depth=1,
    attention_heads=2,
    attention_head_size=16,
    feed_forward_multiplier=1,
    text_embedding_size=16,
    convolution_layers=1,
    ode_solver_method="euler",
)
FEATS_CONF = dict(
    fs=24000,
    n_fft=1024,
    hop_length=256,
    win_length=1024,
    n_mels=100,
    mel_spec_type="vocos",
)


@pytest.fixture
def token_file(tmp_path):
    path = tmp_path / "tokens.txt"
    path.write_text("<blank>\n<unk>\na\nb\n<sos/eos>\n", encoding="utf-8")
    return str(path)


def _build(token_file, **overrides):
    return F5TTS(
        token_list=token_file,
        feats_extract_config=FEATS_CONF,
        **dict(MODEL_CONF, **overrides),
    )


# --------------------------------------------------------------------- solvers


def test_euler_matches_the_closed_form_solution():
    """dy/dt = y from y0 = 1 steps to (1 + dt) ** n under euler."""
    t = torch.linspace(0.0, 1.0, 5)
    sol = odeint(lambda _t, y: y, torch.tensor([1.0]), t, method="euler")

    assert sol.shape == (5, 1)
    torch.testing.assert_close(sol[0], torch.tensor([1.0]))
    torch.testing.assert_close(sol[-1], torch.tensor([1.25**4]))


def test_midpoint_is_second_order_so_beats_euler_on_the_same_grid():
    """Both approximate exp(1); midpoint must land closer."""

    def f(_t, y):
        return y

    t = torch.linspace(0.0, 1.0, 5)
    y0 = torch.tensor([1.0])
    exact = torch.e

    euler_err = abs(odeint(f, y0, t, method="euler")[-1].item() - exact)
    midpoint_err = abs(odeint(f, y0, t, method="midpoint")[-1].item() - exact)

    assert midpoint_err < euler_err


def test_step_size_follows_a_non_uniform_grid():
    """F5's sway sampling produces uneven grids, so dt is read per step."""
    t = torch.tensor([0.0, 0.1, 1.0])
    sol = odeint(lambda _t, y: torch.ones_like(y), torch.tensor([0.0]), t)

    # dy/dt = 1, so the exact solution is y = t regardless of the spacing.
    torch.testing.assert_close(sol.reshape(-1), t)


def test_derivative_receives_the_grid_time_not_the_step_index():
    seen = []
    t = torch.tensor([0.0, 0.25, 0.75])
    odeint(
        lambda ti, y: seen.append(float(ti)) or torch.zeros_like(y),
        torch.tensor([0.0]),
        t,
    )
    assert seen == [0.0, 0.25]


def test_unsupported_method_is_rejected_rather_than_delegated():
    """Adaptive solvers would need torchdiffeq, which espnet does not ship."""
    with pytest.raises(ValueError, match="dopri5"):
        odeint(
            lambda _t, y: y,
            torch.tensor([1.0]),
            torch.linspace(0, 1, 3),
            method="dopri5",
        )


def test_the_error_names_the_supported_methods():
    with pytest.raises(ValueError) as excinfo:
        odeint(lambda _t, y: y, torch.tensor([1.0]), torch.linspace(0, 1, 3), "rk4")
    message = str(excinfo.value)
    assert "euler" in message and "midpoint" in message


# ------------------------------------------------------------------- CFM.sample


@pytest.mark.parametrize("method", ["euler", "midpoint"])
def test_sampling_from_a_raw_reference_waveform(token_file, method):
    """cond as [1, T_wav] makes CFM extract the mel itself, as inference does."""
    model = _build(token_file, ode_solver_method=method)
    cond = torch.randn(1, 24000 // 4)
    text = torch.tensor([[2, 3, 2]])

    out, trajectory = model.cfm.sample(
        cond=cond, text=text, duration=40, steps=2, cfg_strength=2.0
    )

    assert out.shape == (1, 40, 100)
    assert torch.isfinite(out).all()
    # torchdiffeq-compatible: the whole trajectory comes back, one entry per
    # grid point, so trajectory[-1] is the final state.
    assert trajectory.shape[0] > 1


def test_sampling_from_a_precomputed_mel(token_file):
    """cond as [1, n, d] is taken as mel and passed through unchanged."""
    model = _build(token_file)
    cond = torch.randn(1, 20, 100)
    text = torch.tensor([[2, 3]])

    out, _ = model.cfm.sample(cond=cond, text=text, duration=30, steps=2)

    assert out.shape == (1, 30, 100)


def test_seed_makes_sampling_reproducible(token_file):
    model = _build(token_file)
    cond = torch.randn(1, 20, 100)
    text = torch.tensor([[2, 3]])
    kwargs = dict(cond=cond, text=text, duration=30, steps=2, seed=1234)

    first, _ = model.cfm.sample(**kwargs)
    second, _ = model.cfm.sample(**kwargs)

    torch.testing.assert_close(first, second)


# --------------------------------------------------------------- F5TTS.inference


def test_inference_strips_the_reference_prefix(token_file):
    """feat_gen must hold only the generated span, not the prompt."""
    model = _build(token_file)
    ref_mel = torch.randn(1, 20, 100)

    out = model.inference(
        text=torch.tensor([2, 3, 2]), speech=ref_mel, duration=50, steps=2
    )

    # duration 50 total, 20 of which are the reference prefix.
    assert out["feat_gen"].shape == (30, 100)


def test_inference_defaults_duration_to_twice_the_reference(token_file):
    model = _build(token_file)
    ref_mel = torch.randn(1, 16, 100)

    out = model.inference(text=torch.tensor([2, 3]), speech=ref_mel, steps=2)

    assert out["feat_gen"].shape == (16, 100)


def test_inference_accepts_a_raw_reference_waveform(token_file):
    """A [T_wav] reference is passed through to CFM, which extracts its mel."""
    model = _build(token_file)
    ref_wave = torch.randn(24000 // 4)

    out = model.inference(
        text=torch.tensor([2, 3, 2]), speech=ref_wave, duration=40, steps=2
    )

    # No mel length is known up front, so nothing is stripped as a prefix.
    assert out["feat_gen"].shape == (40, 100)


def test_inference_without_a_reference_is_refused(token_file):
    """F5 is zero-shot: there is nothing to clone without reference speech."""
    model = _build(token_file)

    with pytest.raises(RuntimeError, match="reference"):
        model.inference(text=torch.tensor([2, 3]), speech=None)
