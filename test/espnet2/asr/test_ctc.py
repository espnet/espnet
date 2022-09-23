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


@pytest.mark.parametrize("ctc_type", ["builtin", "gtnctc"])
def test_ctc_forward_backward(ctc_type, ctc_args):
    ctc = CTC(encoder_output_size=10, odim=5, ctc_type=ctc_type)
    ctc(*ctc_args).sum().backward()


@pytest.mark.parametrize("ctc_type", ["builtin", "gtnctc"])
def test_ctc_softmax(ctc_type, ctc_args):
    ctc = CTC(encoder_output_size=10, odim=5, ctc_type=ctc_type)
    ctc.softmax(ctc_args[0])


@pytest.mark.parametrize("ctc_type", ["builtin", "gtnctc"])
def test_ctc_log_softmax(ctc_type, ctc_args):
    ctc = CTC(encoder_output_size=10, odim=5, ctc_type=ctc_type)
    ctc.log_softmax(ctc_args[0])


@pytest.mark.parametrize("ctc_type", ["builtin", "gtnctc"])
def test_ctc_argmax(ctc_type, ctc_args):
    ctc = CTC(encoder_output_size=10, odim=5, ctc_type=ctc_type)
    ctc.argmax(ctc_args[0])
