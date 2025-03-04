import pytest
import torch
from packaging.version import parse as V

from espnet2.asr.encoder.hugging_face_transformers_encoder import (
    HuggingFaceTransformersEncoder,
)

is_torch_1_8_plus = V(torch.__version__) >= V("1.8.0")


@pytest.mark.parametrize(
    "model_name_or_path, lang_token_id, encoder_module",
    [
        ("bert-base-uncased", -1, None),
        ("laion/clap-htsat-unfused", -1, "text_model"),
    ],
)
@pytest.mark.execution_timeout(50)
def test_transformers_encoder_forward(
    model_name_or_path, lang_token_id, encoder_module
):
    # get_text_features
    if not is_torch_1_8_plus:
        return

    input_size = 768
    encoder = HuggingFaceTransformersEncoder(
        input_size, model_name_or_path, lang_token_id, encoder_module
    )

    batch_size = 4
    sequence_length = 50
    x = torch.randint(0, 1000, [batch_size, sequence_length])
    x_lengths = torch.LongTensor([20, 5, 50, 15])

    y, y_lengths = encoder(x, x_lengths)
    y.sum().backward()

    odim = encoder.output_size()

    expected_seq_length = sequence_length
    expected_lengths = x_lengths.clone()

    if lang_token_id != -1:
        expected_seq_length += 1
        expected_lengths += 1

    assert y.shape == torch.Size([batch_size, expected_seq_length, odim])
    assert torch.equal(y_lengths, expected_lengths)


@pytest.mark.execution_timeout(30)
def test_reload_pretrained_parameters():
    if not is_torch_1_8_plus:
        return

    input_size = 768
    encoder = HuggingFaceTransformersEncoder(input_size, "akreal/tiny-random-bert")

    saved_param = next(encoder.parameters()).detach().clone()

    next(encoder.parameters()).data *= 0
    new_param = next(encoder.parameters()).detach().clone()

    assert not torch.equal(saved_param, new_param)

    encoder.reload_pretrained_parameters()
    reloaded_param = next(encoder.parameters()).detach().clone()

    assert torch.equal(saved_param, reloaded_param)


@pytest.mark.execution_timeout(30)
def test_output_size():
    if not is_torch_1_8_plus:
        return

    input_size = 768
    encoder = HuggingFaceTransformersEncoder(input_size, "akreal/tiny-random-bert")

    odim = encoder.output_size()
    assert isinstance(odim, int)
    assert odim > 0

    assert odim == encoder.transformer.config.hidden_size


@pytest.mark.execution_timeout(50)
def test_transformers_encoder_invalid_model():
    if not is_torch_1_8_plus:
        return

    input_size = 768
    with pytest.raises(Exception):
        HuggingFaceTransformersEncoder(input_size, "non-existent-model")
