import pytest
import torch

from espnet2.sds.asr.espnet_asr import ESPnetASRModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_forward():
    asr_model = ESPnetASRModel(
        tag=(
            "espnet/"
            "simpleoier_librispeech_asr_train_asr_conformer7_"
            "wavlm_large_raw_en_bpe5000_sp"
        )
    )
    asr_model.warmup()
    x = torch.randn(2000, requires_grad=False)
    asr_model.forward(x)
