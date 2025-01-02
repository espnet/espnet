import pytest
import torch

from espnet2.sds.asr.owsm_ctc_asr import OWSMCTCModel


@pytest.mark.parametrize(
    "tag", ["pyf98/owsm_ctc_v3.1_1B", "espnet/owsm_ctc_v3.2_ft_1B"]
)
def test_forward(tag):
    if not torch.cuda.is_available():
        return  # Only GPU supported
    asr_model = OWSMCTCModel(tag=tag)
    asr_model.warmup()
    x = torch.randn(2000, requires_grad=False)
    asr_model.forward(x)
