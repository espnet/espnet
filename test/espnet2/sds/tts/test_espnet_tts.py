import pytest
import torch

from espnet2.sds.tts.espnet_tts import ESPnetTTSModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
@pytest.mark.parametrize(
    "tag",
    [
        "kan-bayashi/ljspeech_vits",
        "kan-bayashi/libritts_xvector_vits",
        "kan-bayashi/vctk_multi_spk_vits",
    ],
)
def test_forward(tag):
    tts_model = ESPnetTTSModel(tag=tag)
    tts_model.warmup()
    x = "This is dummy sentence"
    tts_model.forward(x)
