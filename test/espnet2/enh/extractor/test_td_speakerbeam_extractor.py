import pytest
import torch

from espnet2.enh.extractor.td_speakerbeam_extractor import TDSpeakerBeamExtractor


@pytest.mark.parametrize("input_dim", [5])
@pytest.mark.parametrize("layer", [4])
@pytest.mark.parametrize("stack", [2])
@pytest.mark.parametrize("bottleneck_dim", [5])
@pytest.mark.parametrize("hidden_dim", [10])
@pytest.mark.parametrize("skip_dim", [5, None])
@pytest.mark.parametrize("kernel", [3])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("norm_type", ["BN", "gLN", "cLN"])
@pytest.mark.parametrize("nonlinear", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("i_adapt_layer", [3])
@pytest.mark.parametrize("adapt_layer_type", ["mul", "concat", "muladd"])
@pytest.mark.parametrize("adapt_enroll_dim", [5])
def test_td_speakerbeam_forward_backward(
    input_dim,
    layer,
    stack,
    bottleneck_dim,
    hidden_dim,
    skip_dim,
    kernel,
    causal,
    norm_type,
    nonlinear,
    i_adapt_layer,
    adapt_layer_type,
    adapt_enroll_dim,
):
    if adapt_layer_type == "muladd":
        adapt_enroll_dim = adapt_enroll_dim * 2
    model = TDSpeakerBeamExtractor(
        input_dim=input_dim,
        layer=layer,
        stack=stack,
        bottleneck_dim=bottleneck_dim,
        hidden_dim=hidden_dim,
        skip_dim=skip_dim,
        kernel=kernel,
        causal=causal,
        norm_type=norm_type,
        nonlinear=nonlinear,
        i_adapt_layer=i_adapt_layer,
        adapt_layer_type=adapt_layer_type,
        adapt_enroll_dim=adapt_enroll_dim,
    )
    model.train()

    x = torch.rand(2, 10, input_dim)
    x_lens = torch.tensor([10, 8], dtype=torch.long)
    enroll = torch.rand(2, 20, input_dim)
    enroll_lens = torch.tensor([20, 18])

    masked, flens, others = model(
        x, ilens=x_lens, input_aux=enroll, ilens_aux=enroll_lens, suffix_tag="_spk1"
    )
    masked.abs().mean().backward()


def test_td_speakerbeam_invalid_type():
    with pytest.raises(ValueError):
        TDSpeakerBeamExtractor(
            input_dim=10,
            nonlinear="fff",
        )
