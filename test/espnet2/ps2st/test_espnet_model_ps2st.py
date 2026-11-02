from types import SimpleNamespace

import numpy as np
import pytest
import torch

from espnet2.ps2st import espnet_model as ps2st_espnet_model
from espnet2.ps2st.espnet_model import ESPnetQwen2AudioModel
from espnet2.text.qwen2audio_tokenizer import Qwen2AudioTokenizer


def _multimodal_inputs(tokenizer):
    inputs = tokenizer.create_multimodal_query(
        text_input="welcome to japari park.",
        audio_input=([np.zeros((16000))], 16000),
    )
    return {key: torch.from_numpy(value) for key, value in inputs.items()}


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
    output = model.inference(**_multimodal_inputs(tokenizer))
    assert output is not None


@pytest.mark.parametrize("model_name", ["Qwen/Qwen2-Audio-7B-Instruct"])
@pytest.mark.execution_timeout(300)
def test_pytest_model_is_built_in_a_single_dtype(model_name):
    """The dummy model must not come out part bfloat16, part float32.

    The checkpoint config asks for bfloat16 and the submodules built through
    `AutoModel.from_config` honour it, but since transformers v5.10 `lm_head`
    and `audio_tower` are constructed on the outer model at the default dtype.
    A mixed model only breaks once the hidden states reach `lm_head`, deep
    inside beam search, so check the dtypes directly where the failure can name
    the tensor that disagrees.
    """
    qwen2audio_model = ESPnetQwen2AudioModel(
        model_name, pytest_mode=True
    ).qwen2audio_model

    tensors = list(qwen2audio_model.named_parameters()) + list(
        qwen2audio_model.named_buffers()
    )
    assert tensors, "no parameters or buffers to check"

    offenders = [
        f"{name}: {tensor.dtype}"
        for name, tensor in tensors
        if tensor.is_floating_point() and tensor.dtype is not torch.float32
    ]
    assert not offenders, f"expected every float tensor to be float32, got {offenders}"


@pytest.mark.parametrize("model_name", ["Qwen/Qwen2-Audio-7B-Instruct"])
@pytest.mark.execution_timeout(300)
def test_inference_takes_token_ids_from_the_text_config(model_name, monkeypatch):
    """sos/eos must be read from the text sub-config.

    They used to come off `qwen2audio_model.language_model.config`, and
    transformers v5.10 moved that attribute to `.model.language_model`. The
    outer `Qwen2AudioConfig` carries no bos/eos of its own, so reaching for the
    ids anywhere but the text config raises instead of quietly decoding with
    the wrong token -- but only once `inference` runs, which is what this pins.
    """
    model = ESPnetQwen2AudioModel(model_name, pytest_mode=True)
    tokenizer = Qwen2AudioTokenizer(model_name)

    recorded = {}

    class RecordingBeamSearch:
        def __init__(self, **kwargs):
            recorded.update(kwargs)

        def __call__(self, *args, **kwargs):
            return [SimpleNamespace(yseq=torch.zeros(1, dtype=torch.long))]

    monkeypatch.setattr(ps2st_espnet_model, "BeamSearch", RecordingBeamSearch)

    model.inference(**_multimodal_inputs(tokenizer))

    text_config = model.qwen2audio_model.config.get_text_config()
    assert isinstance(recorded["sos"], int)
    assert isinstance(recorded["eos"], int)
    assert recorded["sos"] == text_config.bos_token_id
    assert recorded["eos"] == text_config.eos_token_id
