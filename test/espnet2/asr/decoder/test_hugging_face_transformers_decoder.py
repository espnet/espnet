import pytest
import torch

from espnet2.asr.decoder.hugging_face_transformers_decoder import (
    HuggingFaceTransformersDecoder,
)


@pytest.mark.parametrize(
    "model_name_or_path",
    [
        "akreal/tiny-random-t5",
        "akreal/tiny-random-mbart",
    ],
)
@pytest.mark.parametrize("encoder_output_size", [16, 32])
@pytest.mark.execution_timeout(50)
def test_HuggingFaceTransformersDecoder_backward(
    encoder_output_size, model_name_or_path
):
    decoder = HuggingFaceTransformersDecoder(
        vocab_size=5000,  # not used
        encoder_output_size=encoder_output_size,
        model_name_or_path=model_name_or_path,
    )
    x = torch.randn(2, 9, encoder_output_size)
    x_lens = torch.tensor([9, 7], dtype=torch.long)
    t = torch.randint(0, 10, [2, 4], dtype=torch.long)
    t_lens = torch.tensor([4, 3], dtype=torch.long)
    z_all, ys_in_lens = decoder(x, x_lens, t, t_lens)
    z_all.sum().backward()


@pytest.mark.execution_timeout(30)
def test_reload_pretrained_parameters():
    decoder = HuggingFaceTransformersDecoder(5000, 32, "akreal/tiny-random-mbart")
    saved_param = decoder.parameters().__next__().detach().clone()

    decoder.parameters().__next__().data *= 0
    new_param = decoder.parameters().__next__().detach().clone()
    assert not torch.equal(saved_param, new_param)

    decoder.reload_pretrained_parameters()
    new_param = decoder.parameters().__next__().detach().clone()
    assert torch.equal(saved_param, new_param)


@pytest.mark.parametrize(
    "model_name_or_path",
    [
        "akreal/tiny-random-LlamaForCausalLM",  # tokenizer.padding_side=="left"
        "akreal/tiny-random-BloomForCausalLM",  # tokenizer.padding_side=="right"
    ],
)
@pytest.mark.parametrize("prefix", ["prefix", ""])
@pytest.mark.parametrize("postfix", ["postfix", ""])
@pytest.mark.execution_timeout(50)
def test_HuggingFaceTransformersDecoder_causal_lm(model_name_or_path, prefix, postfix):
    encoder_output_size = 32
    decoder = HuggingFaceTransformersDecoder(
        vocab_size=100,
        encoder_output_size=encoder_output_size,
        model_name_or_path=model_name_or_path,
        causal_lm=True,
        prefix=prefix,
        postfix=postfix,
    )
    x = torch.randn(2, 9, encoder_output_size)
    x_lens = torch.tensor([9, 7], dtype=torch.long)
    t = torch.randint(0, 10, [2, 4], dtype=torch.long)
    t_lens = torch.tensor([4, 3], dtype=torch.long)
    z_all, ys_in_lens = decoder(x, x_lens, t, t_lens)
    assert t.shape[1] == z_all.shape[1]
