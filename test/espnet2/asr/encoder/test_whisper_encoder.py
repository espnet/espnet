import pytest
import torch
from packaging.version import parse as V

from espnet2.asr.encoder.whisper_encoder import OpenAIWhisperEncoder

pytest.importorskip("whisper")

# NOTE(Shih-Lun): needed for `return_complex` param in torch.stft()
is_torch_1_7_plus = V(torch.__version__) >= V("1.7.0")


@pytest.fixture()
def whisper_encoder(request):
    encoder = OpenAIWhisperEncoder(whisper_model="tiny")

    return encoder


@pytest.mark.timeout(50)
def test_encoder_init(whisper_encoder):
    if not is_torch_1_7_plus:
        return

    assert whisper_encoder.output_size() == 384


def test_encoder_invalid_init():
    if not is_torch_1_7_plus:
        return

    with pytest.raises(AssertionError):
        encoder = OpenAIWhisperEncoder(whisper_model="aaa")
        del encoder


@pytest.mark.timeout(50)
def test_encoder_forward_no_ilens(whisper_encoder):
    if not is_torch_1_7_plus:
        return

    input_tensor = torch.randn(
        4, 3200, device=next(whisper_encoder.parameters()).device
    )
    xs_pad, _, _ = whisper_encoder(input_tensor, None)

    assert xs_pad.size() == torch.Size([4, 10, 384])


@pytest.mark.timeout(50)
def test_encoder_forward_ilens(whisper_encoder):
    if not is_torch_1_7_plus:
        return

    input_tensor = torch.randn(
        4, 3200, device=next(whisper_encoder.parameters()).device
    )
    input_lens = torch.tensor(
        [500, 1000, 1600, 3200], device=next(whisper_encoder.parameters()).device
    )
    xs_pad, olens, _ = whisper_encoder(input_tensor, input_lens)

    assert xs_pad.size() == torch.Size([4, 10, 384])
    assert torch.equal(olens.cpu(), torch.tensor([2, 3, 5, 10]))


@pytest.mark.timeout(50)
def test_encoder_backward(whisper_encoder):
    if not is_torch_1_7_plus:
        return

    whisper_encoder.train()
    input_tensor = torch.randn(
        4, 3200, device=next(whisper_encoder.parameters()).device
    )
    xs_pad, _, _ = whisper_encoder(input_tensor, None)

    xs_pad.sum().backward()


@pytest.mark.timeout(50)
def test_encoder_padtrim(whisper_encoder):
    if not is_torch_1_7_plus:
        return

    from whisper.audio import N_FRAMES

    whisper_encoder.do_pad_trim = True
    whisper_encoder.eval()

    input_tensor = torch.randn(
        4, 3200, device=next(whisper_encoder.parameters()).device
    )
    xs_pad, _, _ = whisper_encoder(input_tensor, None)
    assert xs_pad.size(1) == N_FRAMES // 2

    input_tensor = torch.randn(
        1, 490000, device=next(whisper_encoder.parameters()).device
    )
    xs_pad, _, _ = whisper_encoder(input_tensor, None)
    assert xs_pad.size(1) == N_FRAMES // 2
