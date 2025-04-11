import torch

from espnet2.sds.asr.owsm_asr import OWSMModel


def test_forward():
    if not torch.cuda.is_available():
        return  # Only GPU supported
    asr_model = OWSMModel()
    asr_model.warmup()
    x = torch.randn(2000, requires_grad=False)
    asr_model.forward(x)
