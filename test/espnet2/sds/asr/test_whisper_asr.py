import pytest
import torch

from espnet2.sds.asr.whisper_asr import WhisperASRModel

pytest.importorskip("whisper")

# NOTE(Shih-Lun): needed for `return_complex` param in torch.stft()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize("tag", ["large", "tiny"])
def test_forward(tag):
    asr_model = WhisperASRModel(tag=tag)
    asr_model.warmup()
    x = torch.randn(2000, requires_grad=False)
    asr_model.forward(x)
