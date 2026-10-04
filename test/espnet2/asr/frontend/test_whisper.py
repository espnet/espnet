import pytest
import torch

from espnet2.asr.frontend.whisper import WhisperFrontend

pytest.importorskip("whisper")

# NOTE(Shih-Lun): required by `return_complex` in torch.stft()


@pytest.fixture()
def whisper_frontend(request):
    with torch.no_grad():
        return WhisperFrontend("tiny")


@pytest.mark.timeout(50)
def test_frontend_init():
    frontend = WhisperFrontend("tiny")
    assert frontend.output_size() == 384


def test_frontend_invalid_init():
    with pytest.raises(AssertionError):
        frontend = WhisperFrontend("aaa")
        del frontend


@pytest.mark.timeout(50)
def test_frontend_forward_no_ilens(whisper_frontend):
    input_tensor = torch.randn(
        4, 3200, device=next(whisper_frontend.parameters()).device
    )
    feats, _ = whisper_frontend(input_tensor, None)

    assert feats.size() == torch.Size([4, 10, 384])


@pytest.mark.timeout(50)
def test_frontend_forward_ilens(whisper_frontend):
    input_tensor = torch.randn(
        4, 3200, device=next(whisper_frontend.parameters()).device
    )
    input_lens = torch.tensor(
        [500, 1000, 1600, 3200], device=next(whisper_frontend.parameters()).device
    )
    feats, feats_lens = whisper_frontend(input_tensor, input_lens)

    assert feats.size() == torch.Size([4, 10, 384])
    assert torch.equal(feats_lens.cpu(), torch.tensor([2, 3, 5, 10]))
