"""Sampling path: ``CFM.sample`` and ``F5TTS.inference``.

The training tests only exercise ``forward``. Everything reached during
generation - the classifier-free-guidance branch, the mel extraction inside
``CFM.sample``, the reference handling in ``F5TTS.inference`` - is covered here
instead. The ODE solver those paths drive is tested in ``test_solvers.py``.
"""

import pytest
import torch

from espnet3.systems.f5_tts.f5_tts import F5TTS

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
    """A [T_wav] reference is measured with CFM's own mel, then stripped."""
    model = _build(token_file)
    ref_wave = torch.randn(24000 // 4)
    n_ref_frames = model.cfm.mel_spec(ref_wave.unsqueeze(0)).shape[-1]

    out = model.inference(
        text=torch.tensor([2, 3, 2]), speech=ref_wave, duration=40, steps=2
    )

    # Only the generated span comes back, exactly as for a mel reference.
    assert out["feat_gen"].shape == (40 - n_ref_frames, 100)


def test_raw_waveform_duration_defaults_to_twice_the_reference(token_file):
    """The fallback must count mel frames, not text tokens."""
    model = _build(token_file)
    ref_wave = torch.randn(24000 // 4)
    n_ref_frames = model.cfm.mel_spec(ref_wave.unsqueeze(0)).shape[-1]

    out = model.inference(text=torch.tensor([2, 3, 2]), speech=ref_wave, steps=2)

    assert out["feat_gen"].shape == (n_ref_frames, 100)


def test_inference_without_a_reference_is_refused(token_file):
    """F5 is zero-shot: there is nothing to clone without reference speech."""
    model = _build(token_file)

    with pytest.raises(RuntimeError, match="reference"):
        model.inference(text=torch.tensor([2, 3]), speech=None)
