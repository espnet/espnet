"""Tests for the ESPnet3 classification model's freeze_param support."""

import pytest
import torch

from espnet2.asr.encoder.transformer_encoder import TransformerEncoder
from espnet2.cls.decoder.linear_decoder import LinearDecoder
from espnet3.systems.esp2_cls.espnet_model import ClassificationModel

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                   | Description                  |
# |---------------------------------------------|------------------------------|
# | test_no_freeze_param_leaves_everything_trainable | The default matches the |
# |                          | ESPnet2 model it subclasses.                    |
# | test_freeze_param_freezes_a_whole_subtree   | A module name freezes every  |
# |                                             | parameter beneath it.        |
# | test_freeze_param_freezes_a_single_parameter | An exact parameter name     |
# |                          | freezes that tensor and nothing else.           |
# | test_freeze_param_is_not_a_prefix_match     | "dec" must not match         |
# |                                             | "decoder.*".                 |
# | test_freeze_param_accepts_several_names     | Every listed name is         |
# |                                             | applied.                     |
# | test_freeze_param_rejects_a_name_that_matches_nothing | A typo fails       |
# |                          | rather than silently training the whole model.  |
# | test_freeze_param_error_lists_the_available_modules | The error names the  |
# |                                             | modules that exist.          |
# | test_freeze_param_is_recorded_on_the_model  | The normalized list is kept  |
# |                                             | as an attribute.             |


def _model(**kwargs):
    """Smallest classification model that still has two frozen-able subtrees."""
    encoder = TransformerEncoder(
        input_size=8,
        output_size=8,
        attention_heads=2,
        linear_units=8,
        num_blocks=1,
        input_layer="linear",
    )
    return ClassificationModel(
        vocab_size=2,
        token_list=["neutral", "happy"],
        frontend=None,
        specaug=None,
        normalize=None,
        preencoder=None,
        encoder=encoder,
        decoder=LinearDecoder(2, encoder_output_size=8),
        **kwargs,
    )


def _trainable(model, prefix):
    return {
        name: param.requires_grad
        for name, param in model.named_parameters()
        if name.startswith(prefix)
    }


def test_no_freeze_param_leaves_everything_trainable():
    """Ensure the default behaviour is the ESPnet2 model's."""
    model = _model()

    assert model.freeze_param == []
    assert all(p.requires_grad for p in model.parameters())


def test_freeze_param_freezes_a_whole_subtree():
    """Ensure a module name freezes every parameter under it."""
    model = _model(freeze_param=["encoder"])

    assert not any(_trainable(model, "encoder.").values())
    assert all(_trainable(model, "decoder.").values())


def test_freeze_param_freezes_a_single_parameter():
    """Ensure an exact parameter name freezes that tensor alone."""
    model = _model(freeze_param=["decoder.linear_out.weight"])

    frozen = _trainable(model, "decoder.")
    assert frozen["decoder.linear_out.weight"] is False
    assert frozen["decoder.linear_out.bias"] is True


def test_freeze_param_is_not_a_prefix_match():
    """Ensure a truncated name does not freeze the module it is a prefix of."""
    with pytest.raises(ValueError, match="matched no parameter: dec"):
        _model(freeze_param=["dec"])


def test_freeze_param_accepts_several_names():
    """Ensure every listed name is applied."""
    model = _model(freeze_param=["encoder", "decoder"])

    assert not any(p.requires_grad for p in model.parameters())


def test_freeze_param_rejects_a_name_that_matches_nothing():
    """Ensure a name matching nothing fails before training starts.

    Such a name is always a typo, and training would otherwise run with
    nothing frozen.
    """
    with pytest.raises(ValueError, match="matched no parameter: frontend.upstream"):
        _model(freeze_param=["frontend.upstream"])


def test_freeze_param_error_lists_the_available_modules():
    """Ensure the error names the modules that could have been meant."""
    with pytest.raises(ValueError, match=r"Top-level modules are: .*'encoder'"):
        _model(freeze_param=["enocder"])


def test_freeze_param_is_recorded_on_the_model():
    """Ensure the names are normalized to a list of str and kept."""
    model = _model(freeze_param=("encoder",))

    assert model.freeze_param == ["encoder"]


@pytest.mark.parametrize("freeze_param", [None, (), []])
def test_freeze_param_accepts_empty_values(freeze_param):
    """Ensure an unset freeze_param never touches requires_grad."""
    model = _model(freeze_param=freeze_param)

    assert model.freeze_param == []
    assert all(p.requires_grad for p in model.parameters())


def test_frozen_parameters_receive_no_gradient():
    """Ensure the freeze survives a backward pass, not just __init__."""
    model = _model(freeze_param=["encoder"])
    speech = torch.randn(2, 20, 8)
    speech_lengths = torch.tensor([20, 15])
    label = torch.tensor([[0], [1]])
    label_lengths = torch.tensor([1, 1])

    loss, _, _ = model(speech, speech_lengths, label, label_lengths)
    loss.backward()

    assert all(p.grad is None for p in model.encoder.parameters())
    assert any(p.grad is not None for p in model.decoder.parameters())
