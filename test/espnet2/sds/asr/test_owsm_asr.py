import pytest
import torch

from espnet2.sds.asr.owsm_asr import OWSMModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_forward():
    asr_model = OWSMModel()
    asr_model.warmup()
    x = torch.randn(2000, requires_grad=False)
    asr_model.forward(x)


class _FakeSpeech2Text:
    """Stand in for Speech2Text and return a fixed decoder hypothesis."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def __call__(self, speech):
        return [
            (
                "<eng><asr><notimestamps> He hoped",
                ["<eng>", "<asr>", "<notimestamps>", "▁He", "▁hoped"],
                [1, 2, 3, 4, 5],
                "He hoped",
                None,
            )
        ]


def _build(monkeypatch, **kwargs):
    from espnet2.bin.s2t_inference import Speech2Text

    monkeypatch.setattr(
        Speech2Text,
        "from_pretrained",
        classmethod(lambda cls, **kw: _FakeSpeech2Text(**kw)),
    )
    return OWSMModel(device="cpu", **kwargs)


def test_forward_drops_the_special_symbols(monkeypatch):
    asr_model = _build(monkeypatch)
    assert asr_model.forward(torch.zeros(1600).numpy()) == "He hoped"


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_dtype_reaches_speech2text(monkeypatch, dtype):
    asr_model = _build(monkeypatch, dtype=dtype)
    assert asr_model.s2t.kwargs["dtype"] == dtype
