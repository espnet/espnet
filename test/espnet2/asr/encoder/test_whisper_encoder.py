import sys

import pytest
import torch
from packaging.version import parse as V

from espnet2.asr.encoder.whisper_encoder import OpenAIWhisperEncoder

pytest.importorskip("whisper")

# NOTE(Shih-Lun): needed for `return_complex` param in torch.stft()
is_torch_1_7_plus = V(torch.__version__) >= V("1.7.0")
is_python_3_8_plus = sys.version_info >= (3, 8)


@pytest.mark.skipif(not is_python_3_8_plus or not is_torch_1_7_plus)
@pytest.fixture()
def whisper_encoder(request):
    encoder = OpenAIWhisperEncoder(whisper_model="tiny")

    return encoder


@pytest.mark.skipif(not is_python_3_8_plus or not is_torch_1_7_plus)
@pytest.mark.timeout(50)
def test_encoder_init(whisper_encoder):
    assert whisper_encoder.output_size() == 384


@pytest.mark.skipif(not is_python_3_8_plus or not is_torch_1_7_plus)
def test_encoder_invalid_init():
    with pytest.raises(AssertionError):
        encoder = OpenAIWhisperEncoder(whisper_model="aaa")
        del encoder


@pytest.mark.skipif(not is_python_3_8_plus or not is_torch_1_7_plus)
@pytest.mark.timeout(50)
def test_encoder_forward_no_ilens(whisper_encoder):
    input_tensor = torch.randn(
        4, 3200, device=next(whisper_encoder.parameters()).device
    )
    xs_pad, _, _ = whisper_encoder(input_tensor, None)

    assert xs_pad.size() == torch.Size([4, 10, 384])


@pytest.mark.skipif(not is_python_3_8_plus or not is_torch_1_7_plus)
@pytest.mark.timeout(50)
def test_encoder_forward_ilens(whisper_encoder):
    input_tensor = torch.randn(
        4, 3200, device=next(whisper_encoder.parameters()).device
    )
    input_lens = torch.tensor(
        [500, 1000, 1600, 3200], device=next(whisper_encoder.parameters()).device
    )
    xs_pad, olens, _ = whisper_encoder(input_tensor, input_lens)

    assert xs_pad.size() == torch.Size([4, 10, 384])
    assert torch.equal(olens.cpu(), torch.tensor([2, 3, 5, 10]))


@pytest.mark.skipif(not is_python_3_8_plus or not is_torch_1_7_plus)
@pytest.mark.timeout(50)
def test_encoder_backward(whisper_encoder):
    whisper_encoder.train()
    input_tensor = torch.randn(
        4, 3200, device=next(whisper_encoder.parameters()).device
    )
    xs_pad, _, _ = whisper_encoder(input_tensor, None)

    xs_pad.sum().backward()
