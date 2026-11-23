import pytest
import torch

from espnet2.sds.asr.owsm_ctc_asr import OWSMCTCModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize(
    "tag", ["pyf98/owsm_ctc_v3.1_1B", "espnet/owsm_ctc_v3.2_ft_1B"]
)
def test_forward(tag):
    asr_model = OWSMCTCModel(tag=tag)
    asr_model.warmup()
    x = torch.randn(2000, requires_grad=False)
    asr_model.forward(x)


class _FakeSpeech2Text:
    """Stand in for Speech2Text and return a fixed best-path hypothesis."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs

    def best_path(self, speech):
        # The CTC path writes no language or task symbol, so text and
        # text_nospecial are the same string.
        return [("He hoped", ["▁He", "▁hoped"], [1, 2], "He hoped", None)]


def _build(monkeypatch, **kwargs):
    from espnet2.bin.s2t_inference import Speech2Text

    monkeypatch.setattr(
        Speech2Text,
        "from_pretrained",
        classmethod(lambda cls, tag, **kw: _FakeSpeech2Text(**kw)),
    )
    return OWSMCTCModel(device="cpu", **kwargs)


def test_forward_keeps_the_first_word(monkeypatch):
    asr_model = _build(monkeypatch)
    assert asr_model.forward(torch.zeros(1600).numpy()) == "He hoped"


@pytest.mark.parametrize("dtype", ["float32", "float16"])
def test_dtype_reaches_speech2text(monkeypatch, dtype):
    asr_model = _build(monkeypatch, dtype=dtype)
    assert asr_model.s2t.kwargs["dtype"] == dtype
