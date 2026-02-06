import pytest
import torch
from packaging.version import parse as V

from espnet2.asr.decoder.hugging_face_transformers_decoder import (
    HuggingFaceTransformersDecoder,
    read_json_config,
)

is_torch_2_6_plus = V(torch.__version__) >= V("2.6.0")


@pytest.mark.skipif(not is_torch_2_6_plus, reason="Require torch 2.6.0+")
@pytest.mark.parametrize(
    "model_name_or_path",
    [
        "hf-internal-testing/tiny-random-T5Model",
        "hf-internal-testing/tiny-random-MBartModel",
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


@pytest.mark.skipif(not is_torch_2_6_plus, reason="Require torch 2.6.0+")
@pytest.mark.execution_timeout(30)
def test_reload_pretrained_parameters():
    decoder = HuggingFaceTransformersDecoder(
        5000, 32, "hf-internal-testing/tiny-random-MBartModel"
    )
    saved_param = decoder.parameters().__next__().detach().clone()

    decoder.parameters().__next__().data *= 0
    new_param = decoder.parameters().__next__().detach().clone()
    assert not torch.equal(saved_param, new_param)

    decoder.reload_pretrained_parameters()
    new_param = decoder.parameters().__next__().detach().clone()
    assert torch.equal(saved_param, new_param)


@pytest.mark.skipif(not is_torch_2_6_plus, reason="Require torch 2.6.0+")
@pytest.mark.execution_timeout(30)
def test_skip_reload_pretrained_parameters():
    decoder = HuggingFaceTransformersDecoder(
        5000,
        32,
        "hf-internal-testing/tiny-random-MBartModel",
        load_pretrained_weights=False,
    )
    # 1. Parameters loaded from hf in init should not be zero.
    # 2. After zeroing them out they should remain zero post reloading.
    saved_param = decoder.parameters().__next__().detach().clone()
    decoder.parameters().__next__().data *= 0
    zeroed_param = decoder.parameters().__next__().detach().clone()
    assert not torch.equal(saved_param, zeroed_param)

    decoder.reload_pretrained_parameters()
    param_after_maybe_reloading = decoder.parameters().__next__().detach().clone()
    assert torch.equal(
        param_after_maybe_reloading, zeroed_param
    )  # Reloading should be skipped


@pytest.mark.skipif(not is_torch_2_6_plus, reason="Require torch 2.6.0+")
@pytest.mark.execution_timeout(30)
def test_override_hf_decoder_config():
    overriding_architecture_config = {"d_model": 8, "ignore_mismatched_sizes": True}
    decoder = HuggingFaceTransformersDecoder(
        5000,
        32,
        "hf-internal-testing/tiny-random-MBartModel",
        overriding_architecture_config=overriding_architecture_config,
    )
    assert decoder.decoder.embed_tokens.weight.shape == (5000, 8)

    # Check that without overriding embedding matrix has shape=(5000, 32)
    decoder = HuggingFaceTransformersDecoder(
        5000,
        32,
        "hf-internal-testing/tiny-random-MBartModel",
    )
    assert decoder.decoder.embed_tokens.weight.shape == (5000, 16)


@pytest.mark.skipif(not is_torch_2_6_plus, reason="Require torch 2.6.0+")
@pytest.mark.parametrize(
    "model_name_or_path",
    [
        # test a model with tokenizer.padding_side=="left"
        "hf-internal-testing/tiny-random-LlamaForCausalLM",
        # test a model with tokenizer.padding_side=="right"
        "hf-internal-testing/tiny-random-BloomForCausalLM",
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


@pytest.fixture()
def json_config_path(tmp_path):
    json_config = tmp_path / "config.json"
    json_config.write_text(
        """
        {
            "model_name_or_path": "hf-internal-testing/tiny-random-T5Model",
            "encoder_output_size": 32,
            "causal_lm": true,
            "prefix": "prefix",
            "postfix": "postfix"
        }
        """
    )
    return str(json_config)


def test_read_json_config(json_config_path):
    config = read_json_config(json_config_path)
    assert config["model_name_or_path"] == "hf-internal-testing/tiny-random-T5Model"
    assert config["encoder_output_size"] == 32
    assert config["causal_lm"]
    assert config["prefix"] == "prefix"
    assert config["postfix"] == "postfix"
