import numpy as np
import pytest

from espnet2.text.qwen2audio_tokenizer import Qwen2AudioTokenizer


@pytest.mark.parametrize("model_name", ["Qwen/Qwen2-Audio-7B-Instruct"])
def test_qwen2audio_tokenizer(model_name):
    tokenizer = Qwen2AudioTokenizer(model_name)
    assert tokenizer is not None

    text = "welcome to japari park."

    tokens = tokenizer.text2tokens(text)

    rec_text = tokenizer.tokens2text(tokens)

    assert text == rec_text

    # text-only
    output = tokenizer.create_multimodal_query(text)
    assert output is not None

    # text + speech
    speech = np.zeros((16000))
    output = tokenizer.create_multimodal_query(
        text_input=text, audio_input=([speech], 16000)
    )
    assert output is not None
