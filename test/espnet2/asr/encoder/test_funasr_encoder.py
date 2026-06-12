"""Unit tests for FunASREncoder.

These tests use a small synthetic FunASR encoder (built entirely from FunASR's
publicly-registered classes) so that no network download is required during
CI.  The tests mock ``funasr.AutoModel.build_model`` to return a lightweight
in-process model.
"""

import copy
from contextlib import contextmanager
from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

pytest.importorskip("funasr")

from espnet2.asr.encoder.funasr_encoder import FunASREncoder  # noqa: E402


# ---------------------------------------------------------------------------
# Minimal stub that mimics the shape of a real FunASR model with an encoder
# and an optional normalize sub-module.
# ---------------------------------------------------------------------------


class _DummyNormalize(nn.Module):
    """Identity normalizer that matches the FunASR normalize API."""

    def forward(self, xs_pad, ilens):
        return xs_pad, ilens


class _DummyEncoder(nn.Module):
    """Tiny linear encoder that returns a 2-tuple (as SenseVoice does)."""

    def __init__(self, idim=80, odim=64):
        super().__init__()
        self.proj = nn.Linear(idim, odim)
        self._output_size = odim

    def output_size(self) -> int:
        return self._output_size

    def forward(self, xs_pad, ilens):
        return self.proj(xs_pad), ilens


class _DummyEncoderThreeTuple(nn.Module):
    """Tiny encoder that returns a 3-tuple (as SANMEncoder/Paraformer does)."""

    def __init__(self, idim=80, odim=64):
        super().__init__()
        self.proj = nn.Linear(idim, odim)
        self._output_size = odim

    def output_size(self) -> int:
        return self._output_size

    def forward(self, xs_pad, ilens):
        return self.proj(xs_pad), ilens, None


class _DummyFunASRModel(nn.Module):
    def __init__(self, encoder, normalize=None):
        super().__init__()
        self.encoder = encoder
        self.normalize = normalize

    def eval(self):
        return self


def _make_build_model(encoder, normalize=None):
    """Return a replacement for ``AutoModel.build_model`` that returns a stub."""

    def _build_model(**kwargs):
        model = _DummyFunASRModel(encoder=encoder, normalize=normalize)
        return model, kwargs

    return _build_model


# ---------------------------------------------------------------------------
# Helper: patch both AutoModel and is_funasr_available at once so that
# FunASREncoder.__init__ uses our stub regardless of installation status.
# ---------------------------------------------------------------------------


@contextmanager
def _mock_funasr(encoder, normalize=None):
    """Patch the module-level AutoModel and mark funasr as available."""
    mock_cls = type(
        "_MockAutoModel",
        (),
        {"build_model": staticmethod(_make_build_model(encoder, normalize))},
    )
    with patch("espnet2.asr.encoder.funasr_encoder.AutoModel", mock_cls), patch(
        "espnet2.asr.encoder.funasr_encoder.is_funasr_available", True
    ):
        yield


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.timeout(30)
def test_output_size_two_tuple():
    enc = _DummyEncoder(idim=80, odim=64)
    with _mock_funasr(enc):
        wrapper = FunASREncoder(
            input_size=80,
            model_name_or_path="dummy/model",
            hub="ms",
            use_normalize=False,
        )
    assert wrapper.output_size() == 64


@pytest.mark.timeout(30)
def test_output_size_three_tuple():
    enc = _DummyEncoderThreeTuple(idim=80, odim=128)
    with _mock_funasr(enc):
        wrapper = FunASREncoder(
            input_size=80,
            model_name_or_path="dummy/model",
            hub="ms",
            use_normalize=False,
        )
    assert wrapper.output_size() == 128


@pytest.mark.timeout(30)
def test_forward_two_tuple():
    enc = _DummyEncoder(idim=80, odim=64)
    with _mock_funasr(enc):
        wrapper = FunASREncoder(
            input_size=80,
            model_name_or_path="dummy/model",
            hub="ms",
            use_normalize=False,
        )
    xs = torch.randn(2, 20, 80)
    ilens = torch.tensor([20, 15])
    out, olens, hidden = wrapper(xs, ilens)

    assert out.shape == (2, 20, 64)
    assert olens.tolist() == [20, 15]
    assert hidden is None


@pytest.mark.timeout(30)
def test_forward_three_tuple():
    enc = _DummyEncoderThreeTuple(idim=80, odim=128)
    with _mock_funasr(enc):
        wrapper = FunASREncoder(
            input_size=80,
            model_name_or_path="dummy/model",
            hub="ms",
            use_normalize=False,
        )
    xs = torch.randn(2, 30, 80)
    ilens = torch.tensor([30, 20])
    out, olens, hidden = wrapper(xs, ilens)

    assert out.shape == (2, 30, 128)
    assert hidden is None


@pytest.mark.timeout(30)
def test_forward_with_normalize():
    enc = _DummyEncoder(idim=80, odim=64)
    norm = _DummyNormalize()
    with _mock_funasr(enc, normalize=norm):
        wrapper = FunASREncoder(
            input_size=80,
            model_name_or_path="dummy/model",
            hub="ms",
            use_normalize=True,
        )
    xs = torch.randn(2, 20, 80)
    ilens = torch.tensor([20, 15])
    out, olens, _ = wrapper(xs, ilens)
    assert out.shape == (2, 20, 64)


@pytest.mark.timeout(30)
def test_forward_normalize_skipped_when_none():
    """use_normalize=True but model has no normalize → should work fine."""
    enc = _DummyEncoder(idim=80, odim=64)
    with _mock_funasr(enc, normalize=None):
        wrapper = FunASREncoder(
            input_size=80,
            model_name_or_path="dummy/model",
            hub="ms",
            use_normalize=True,
        )
    assert wrapper.funasr_normalize is None
    xs = torch.randn(2, 20, 80)
    ilens = torch.tensor([20, 15])
    out, _, _ = wrapper(xs, ilens)
    assert out.shape == (2, 20, 64)


@pytest.mark.timeout(30)
def test_freeze_encoder():
    enc = _DummyEncoder(idim=80, odim=64)
    with _mock_funasr(enc):
        wrapper = FunASREncoder(
            input_size=80,
            model_name_or_path="dummy/model",
            hub="ms",
            use_normalize=False,
            freeze_encoder=True,
        )
    for param in wrapper.funasr_encoder.parameters():
        assert not param.requires_grad


@pytest.mark.timeout(30)
def test_reload_pretrained_parameters():
    enc = _DummyEncoder(idim=80, odim=64)
    with _mock_funasr(enc):
        wrapper = FunASREncoder(
            input_size=80,
            model_name_or_path="dummy/model",
            hub="ms",
            use_normalize=False,
        )

    # Corrupt the encoder weights.
    with torch.no_grad():
        for p in wrapper.funasr_encoder.parameters():
            p.fill_(99.0)

    wrapper.reload_pretrained_parameters()

    for name, p in wrapper.funasr_encoder.named_parameters():
        expected = wrapper._pretrained_params[name]
        assert torch.allclose(p, expected), f"Parameter '{name}' was not reloaded."


@pytest.mark.timeout(30)
def test_backward_pass():
    enc = _DummyEncoder(idim=80, odim=64)
    with _mock_funasr(enc):
        wrapper = FunASREncoder(
            input_size=80,
            model_name_or_path="dummy/model",
            hub="ms",
            use_normalize=False,
        )
    wrapper.train()
    xs = torch.randn(2, 20, 80, requires_grad=True)
    ilens = torch.tensor([20, 15])
    out, _, _ = wrapper(xs, ilens)
    out.sum().backward()
    assert xs.grad is not None


@pytest.mark.timeout(30)
def test_missing_funasr_raises():
    """Ensure a helpful ImportError is raised when funasr is unavailable."""
    with patch("espnet2.asr.encoder.funasr_encoder.is_funasr_available", False):
        with pytest.raises(ImportError, match="funasr"):
            FunASREncoder(
                input_size=80,
                model_name_or_path="dummy/model",
            )


@pytest.mark.timeout(30)
def test_model_without_encoder_raises():
    """A FunASR model that has no 'encoder' attribute should raise RuntimeError."""

    class _NoEncoderModel(nn.Module):
        def eval(self):
            return self

    def _bad_build(**kwargs):
        return _NoEncoderModel(), kwargs

    mock_cls = type("_MockAutoModel", (), {"build_model": staticmethod(_bad_build)})
    with patch("espnet2.asr.encoder.funasr_encoder.AutoModel", mock_cls), patch(
        "espnet2.asr.encoder.funasr_encoder.is_funasr_available", True
    ):
        with pytest.raises(RuntimeError, match="encoder"):
            FunASREncoder(
                input_size=80,
                model_name_or_path="dummy/model",
            )

