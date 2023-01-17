import pytest
import torch

from espnet2.asr.encoder.whisper_encoder import OpenAIWhisperEncoder

pytest.importorskip("whisper")


@pytest.fixture()
def whisper_encoder(request):
    encoder = OpenAIWhisperEncoder()
    encoder.train()

    return encoder


@pytest.mark.timeout(50)
def test_encoder_init(whisper_encoder):
    assert whisper_encoder.output_size() == 768


def test_encoder_invalid_init():
    with pytest.raises(AssertionError):
        encoder = OpenAIWhisperEncoder(whisper_model="aaa")
        del encoder


@pytest.mark.timeout(50)
def test_encoder_forward_no_ilens(whisper_encoder):
    input_tensor = torch.randn(
        4, 32000, device=next(whisper_encoder.parameters()).device
    )
    xs_pad, _, _ = whisper_encoder(input_tensor, None)

    assert xs_pad.size() == torch.Size([4, 100, 768])


@pytest.mark.timeout(50)
def test_encoder_forward_ilens(whisper_encoder):
    input_tensor = torch.randn(
        4, 32000, device=next(whisper_encoder.parameters()).device
    )
    input_lens = torch.tensor(
        [5000, 10000, 16000, 32000], device=next(whisper_encoder.parameters()).device
    )
    xs_pad, olens, _ = whisper_encoder(input_tensor, input_lens)

    assert xs_pad.size() == torch.Size([4, 100, 768])
    assert torch.equal(olens.cpu(), torch.tensor([16, 31, 50, 100]))


@pytest.mark.timeout(50)
def test_encoder_backward(whisper_encoder):
    input_tensor = torch.randn(
        4, 32000, device=next(whisper_encoder.parameters()).device
    )
    xs_pad, _, _ = whisper_encoder(input_tensor, None)

    xs_pad.sum().backward()


@pytest.mark.timeout(50)
def test_encoder_padtrim(whisper_encoder):
    from whisper.audio import N_FRAMES

    whisper_encoder.do_pad_trim = True

    input_tensor = torch.randn(
        4, 32000, device=next(whisper_encoder.parameters()).device
    )
    xs_pad, _, _ = whisper_encoder(input_tensor, None)
    assert xs_pad.size(1) == N_FRAMES // 2

    input_tensor = torch.randn(
        4, 540000, device=next(whisper_encoder.parameters()).device
    )
    xs_pad, _, _ = whisper_encoder(input_tensor, None)
    assert xs_pad.size(1) == N_FRAMES // 2
