import pytest
import torch

from espnet2.asr.ctc import CTC


@pytest.fixture
def ctc_args():
    bs = 2
    h = torch.randn(bs, 10, 10)
    h_lens = torch.LongTensor([10, 8])
    y = torch.randint(0, 4, [2, 5])
    y_lens = torch.LongTensor([5, 2])
    return h, h_lens, y, y_lens


@pytest.mark.parametrize("ctc_type", ["builtin"])
def test_ctc_forward_backward(ctc_type, ctc_args):
    ctc = CTC(encoder_output_size=10, odim=5, ctc_type=ctc_type)
    ctc(*ctc_args).sum().backward()


@pytest.mark.parametrize("ctc_type", ["builtin"])
def test_ctc_softmax(ctc_type, ctc_args):
    ctc = CTC(encoder_output_size=10, odim=5, ctc_type=ctc_type)
    ctc.softmax(ctc_args[0])


@pytest.mark.parametrize("ctc_type", ["builtin"])
def test_ctc_log_softmax(ctc_type, ctc_args):
    ctc = CTC(encoder_output_size=10, odim=5, ctc_type=ctc_type)
    ctc.log_softmax(ctc_args[0])


@pytest.mark.parametrize("ctc_type", ["builtin"])
def test_ctc_argmax(ctc_type, ctc_args):
    ctc = CTC(encoder_output_size=10, odim=5, ctc_type=ctc_type)
    ctc.argmax(ctc_args[0])


def test_bayes_risk_ctc(ctc_args):
    # Skip the test if K2 is not installed
    try:
        import k2  # noqa
    except ImportError:
        return

    builtin_ctc = CTC(encoder_output_size=10, odim=5, ctc_type="builtin")
    bayes_risk_ctc = CTC(encoder_output_size=10, odim=5, ctc_type="brctc")
    bayes_risk_ctc.ctc_lo = builtin_ctc.ctc_lo

    builtin_ctc_loss = builtin_ctc(*ctc_args)
    bayes_risk_ctc_loss = bayes_risk_ctc(*ctc_args)

    assert torch.abs(builtin_ctc_loss - bayes_risk_ctc_loss) < 1e-6


def test_ctc_forced_align(ctc_args):
    _ = pytest.importorskip("torchaudio")

    ctc = CTC(encoder_output_size=10, odim=5, ctc_type="builtin")
    hs_pad, hlens, ys_pad, ys_lens = ctc_args
    # Forced alignment only works with batch size 1.
    hs_pad = hs_pad[0:1, :]
    hlens = hlens[0:1]
    ys_pad = ys_pad[0:1, :]
    ys_lens = ys_lens[0:1]
    b, t, _ = hs_pad.shape
    blank_idx = 0
    ys_pad[ys_pad == blank_idx] = (
        1  # make sure there is no blank in the target sequence
    )
    aligns, scores = ctc.forced_align(
        hs_pad=hs_pad, hlens=hlens, ys_pad=ys_pad, ys_lens=ys_lens, blank_idx=blank_idx
    )
    assert aligns.shape == (b, t)
    assert scores.shape == (b, t)
