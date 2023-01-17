import pytest
import torch

from espnet2.asr.frontend.whisper import WhisperFrontend

pytest.importorskip("whisper")


@pytest.fixture()
def whisper_frontend(request):
    with torch.no_grad():
        return WhisperFrontend()


@pytest.mark.timeout(50)
def test_frontend_init():
    frontend = WhisperFrontend()
    assert frontend.output_size() == 768


def test_frontend_invalid_init():
    with pytest.raises(AssertionError):
        frontend = WhisperFrontend("aaa")
        del frontend


@pytest.mark.timeout(50)
def test_frontend_forward_no_ilens(whisper_frontend):
    input_tensor = torch.randn(
        4, 32000, device=next(whisper_frontend.parameters()).device
    )
    feats, _ = whisper_frontend(input_tensor, None)

    assert feats.size() == torch.Size([4, 100, 768])


@pytest.mark.timeout(50)
def test_frontend_forward_ilens(whisper_frontend):
    input_tensor = torch.randn(
        4, 32000, device=next(whisper_frontend.parameters()).device
    )
    input_lens = torch.tensor(
        [5000, 10000, 16000, 32000], device=next(whisper_frontend.parameters()).device
    )
    feats, feats_lens = whisper_frontend(input_tensor, input_lens)

    assert feats.size() == torch.Size([4, 100, 768])
    assert torch.equal(feats_lens.cpu(), torch.tensor([16, 31, 50, 100]))
