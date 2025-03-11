import torch

from espnet2.sds.tts.chat_tts import ChatTTSModel


def test_forward():
    if not torch.cuda.is_available():
        return  # Only GPU supported
    tts_model = ChatTTSModel()
    tts_model.warmup()
    x = "This is dummy sentence"
    tts_model.forward(x)
