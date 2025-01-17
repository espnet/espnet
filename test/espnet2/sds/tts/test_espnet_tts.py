import pytest
import torch

from espnet2.sds.tts.espnet_tts import ESPnetTTSModel


@pytest.mark.parametrize(
    "tag",
    [
        "kan-bayashi/ljspeech_vits",
        "kan-bayashi/libritts_xvector_vits",
        "kan-bayashi/vctk_multi_spk_vits",
    ],
)
def test_forward(tag):
    if not torch.cuda.is_available():
        return  # Only GPU supported
    tts_model = ESPnetTTSModel(tag=tag)
    tts_model.warmup()
    x = "This is dummy sentence"
    tts_model.forward(x)
