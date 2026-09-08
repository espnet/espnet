import pytest
import torch

from espnet2.sds.tts.chat_tts import ChatTTSModel


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a GPU")
def test_forward():
    tts_model = ChatTTSModel()
    tts_model.warmup()
    x = "This is dummy sentence"
    tts_model.forward(x)
