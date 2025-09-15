import numpy as np
import pytest
import torch

from espnet2.ps2st.espnet_model import ESPnetQwen2AudioModel
from espnet2.text.qwen2audio_tokenizer import Qwen2AudioTokenizer


@pytest.mark.parametrize("model_name", ["Qwen/Qwen2-Audio-7B-Instruct"])
@pytest.mark.execution_timeout(300)
def test_espnet_model_inference(model_name):
    model = ESPnetQwen2AudioModel(model_name, pytest_mode=True)
    tokenizer = Qwen2AudioTokenizer(model_name)
    assert model is not None
    assert tokenizer is not None
    model.decode_config["maxlenratio"] = 1.0

    # We don't test forward function because it's a dummy function here.
    # Instead, we test the inference function.
    # Qwen2 scorer is also tested within inference function.
    text = "welcome to japari park."
    speech = np.zeros((16000))
    inputs = tokenizer.create_multimodal_query(
        text_input=text, audio_input=([speech], 16000)
    )

    for key in inputs.keys():
        inputs[key] = torch.from_numpy(inputs[key])

    output = model.inference(**inputs)
    assert output is not None
