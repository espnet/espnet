import pytest
import torch
from packaging.version import parse as V

from espnet2.asr.postencoder.hugging_face_transformers_postencoder import (
    HuggingFaceTransformersPostEncoder,
)

is_torch_2_6_plus = V(torch.__version__) >= V("2.6.0")


@pytest.mark.parametrize(
    "model_name_or_path, length_adaptor_n_layers, lang_token_id",
    [
        ("hf-internal-testing/tiny-random-BertModel", 0, 1),
        ("hf-internal-testing/tiny-random-GPT2Model", 0, 1),
        ("hf-internal-testing/tiny-random-XLNetModel", 0, 1),
        ("hf-internal-testing/tiny-random-T5Model", 0, 1),
        ("hf-internal-testing/tiny-random-MBartModel", 0, 1),
        ("hf-internal-testing/tiny-random-MBartModel", 0, -1),
        ("hf-internal-testing/tiny-random-MBartModel", 1, -1),
        ("hf-internal-testing/tiny-random-MPNetModel", 0, 1),
    ],
)
@pytest.mark.execution_timeout(50)
def test_transformers_forward(
    model_name_or_path, length_adaptor_n_layers, lang_token_id
):
    if not is_torch_2_6_plus:
        return
    idim = 400
    postencoder = HuggingFaceTransformersPostEncoder(
        idim, model_name_or_path, length_adaptor_n_layers, lang_token_id
    )
    x = torch.randn([4, 50, idim], requires_grad=True)
    x_lengths = torch.LongTensor([20, 5, 50, 15])
    y, y_lengths = postencoder(x, x_lengths)
    y.sum().backward()
    odim = postencoder.output_size()

    y_shape_1_expected = 50 // 2**length_adaptor_n_layers
    y_lengths_expected = (
        x_lengths.float().div(2**length_adaptor_n_layers).floor().long()
    )

    if lang_token_id != -1:
        y_shape_1_expected += 1
        y_lengths_expected += 1

    assert y.shape == torch.Size([4, y_shape_1_expected, odim])
    assert torch.equal(y_lengths, y_lengths_expected)


@pytest.mark.execution_timeout(30)
def test_transformers_too_short_utt():
    if not is_torch_2_6_plus:
        return
    idim = 400
    postencoder = HuggingFaceTransformersPostEncoder(
        idim, "hf-internal-testing/tiny-random-BertModel", 2
    )
    x = torch.randn([2, 3, idim], requires_grad=True)
    x_lengths = torch.LongTensor([3, 2])
    with pytest.raises(Exception):
        y, y_lengths = postencoder(x, x_lengths)


@pytest.mark.execution_timeout(30)
def test_reload_pretrained_parameters():
    if not is_torch_2_6_plus:
        return
    postencoder = HuggingFaceTransformersPostEncoder(
        400, "hf-internal-testing/tiny-random-BertModel"
    )
    saved_param = postencoder.parameters().__next__().detach().clone()

    postencoder.parameters().__next__().data *= 0
    new_param = postencoder.parameters().__next__().detach().clone()
    assert not torch.equal(saved_param, new_param)

    postencoder.reload_pretrained_parameters()
    new_param = postencoder.parameters().__next__().detach().clone()
    assert torch.equal(saved_param, new_param)
