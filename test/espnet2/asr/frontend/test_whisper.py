import sys

import pytest
import torch
from packaging.version import parse as V

from espnet2.asr.frontend.whisper import WhisperFrontend

pytest.importorskip("whisper")

# NOTE(Shih-Lun): required by `return_complex` in torch.stft()
is_torch_1_7_plus = V(torch.__version__) >= V("1.7.0")
is_python_3_8_plus = sys.version_info >= (3, 8)


@pytest.mark.skipif(
    not is_python_3_8_plus or not is_torch_1_7_plus,
    reason="whisper not supported on python<3.8, torch<1.7",
)
@pytest.fixture()
def whisper_frontend(request):
    with torch.no_grad():
        return WhisperFrontend("tiny")


@pytest.mark.skipif(
    not is_python_3_8_plus or not is_torch_1_7_plus,
    reason="whisper not supported on python<3.8, torch<1.7",
)
@pytest.mark.timeout(50)
def test_frontend_init():
    frontend = WhisperFrontend("tiny")
    assert frontend.output_size() == 384


@pytest.mark.skipif(
    not is_python_3_8_plus or not is_torch_1_7_plus,
    reason="whisper not supported on python<3.8, torch<1.7",
)
def test_frontend_invalid_init():
    with pytest.raises(AssertionError):
        frontend = WhisperFrontend("aaa")
        del frontend


@pytest.mark.skipif(
    not is_python_3_8_plus or not is_torch_1_7_plus,
    reason="whisper not supported on python<3.8, torch<1.7",
)
@pytest.mark.timeout(50)
def test_frontend_forward_no_ilens(whisper_frontend):
    input_tensor = torch.randn(
        4, 3200, device=next(whisper_frontend.parameters()).device
    )
    feats, _ = whisper_frontend(input_tensor, None)

    assert feats.size() == torch.Size([4, 10, 384])


@pytest.mark.skipif(
    not is_python_3_8_plus or not is_torch_1_7_plus,
    reason="whisper not supported on python<3.8, torch<1.7",
)
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
