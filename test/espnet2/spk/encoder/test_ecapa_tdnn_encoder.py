import pytest
import torch

from espnet2.spk.encoder.ecapa_tdnn_encoder import EcapaTdnnEncoder


def _encoder(**kwargs):
    return EcapaTdnnEncoder(
        input_size=20, model_scale=2, ndim=16, output_size=24, **kwargs
    )


def _dilations(encoder):
    blocks = [encoder.layer1, encoder.layer2, encoder.layer3, *encoder.extra_layers]
    return [block.convs[0].dilation[0] for block in blocks]


def test_default_is_the_three_block_encoder():
    torch.manual_seed(0)
    encoder = _encoder().eval()
    x = torch.randn(2, 50, 20)

    assert not any(key.startswith("extra_layers") for key in encoder.state_dict())
    assert encoder.layer4.weight.shape == (24, 3 * 16, 1)
    assert _dilations(encoder) == [2, 3, 4]
    # The forward pass written out as it was before blocks could be added.
    with torch.no_grad():
        h = encoder.bn(encoder.relu(encoder.conv(x.permute(0, 2, 1))))
        x1 = encoder.layer1(h)
        x2 = encoder.layer2(h + x1)
        x3 = encoder.layer3(h + x1 + x2)
        expected = encoder.relu(encoder.layer4(torch.cat((x1, x2, x3), dim=1)))
        torch.testing.assert_close(encoder(x), expected)


def test_num_blocks_and_dilations_build_the_same_model():
    by_count = _encoder(num_blocks=4).eval()
    by_dilations = _encoder(dilations=[2, 3, 4, 5]).eval()
    by_dilations.load_state_dict(by_count.state_dict())
    x = torch.randn(2, 50, 20)

    assert _dilations(by_count) == _dilations(by_dilations) == [2, 3, 4, 5]
    with torch.no_grad():
        torch.testing.assert_close(by_count(x), by_dilations(x))


@pytest.mark.parametrize("dilations", [[2, 3, 4], [2, 3, 4, 5], [2, 3, 4, 2, 3, 4]])
def test_dilations_set_the_blocks(dilations):
    encoder = _encoder(dilations=dilations)

    assert _dilations(encoder) == dilations
    assert len(encoder.extra_layers) == len(dilations) - 3
    assert encoder.layer4.in_channels == len(dilations) * 16
    assert encoder(torch.randn(2, 50, 20)).shape == (2, 24, 50)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_blocks": 2},
        {"dilations": [2, 3]},
        {"num_blocks": 4, "dilations": [2, 3, 4]},
    ],
)
def test_rejects_too_few_or_mismatched_blocks(kwargs):
    with pytest.raises(ValueError):
        _encoder(**kwargs)
