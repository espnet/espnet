import sys

import pytest
import torch
from packaging.version import parse as V

from espnet2.sds.asr.whisper_asr import WhisperASRModel

pytest.importorskip("whisper")

# NOTE(Shih-Lun): needed for `return_complex` param in torch.stft()
is_torch_1_7_plus = V(torch.__version__) >= V("1.7.0")
is_python_3_8_plus = sys.version_info >= (3, 8)


@pytest.mark.skipif(
    not is_python_3_8_plus or not is_torch_1_7_plus,
    reason="whisper not supported on python<3.8, torch<1.7",
)
@pytest.mark.parametrize("tag", ["large", "tiny"])
def test_forward(tag):
    if not torch.cuda.is_available():
        return  # Only GPU supported
    asr_model = WhisperASRModel(tag=tag)
    asr_model.warmup()
    x = torch.randn(2000, requires_grad=False)
    asr_model.forward(x)
