import pytest
import torch

from espnet2.spk.encoder.ecapa_tdnn_encoder import EcapaTdnnEncoder

# Output [0, :4, :4] of the encoder before num_blocks and dilations were added,
# built after torch.manual_seed(0) and run on the next torch.randn(2, 50, 20).
BEFORE_THIS_CHANGE = torch.tensor(
    [
        [0.96268100, 1.22469091, 1.90425205, 1.42317164],
        [0.60476065, 0.20689973, 0.83711189, 0.20918621],
        [0.07511241, 0.07382777, 0.11325108, 0.19168153],
        [0.05349822, 0.00357660, 0.06722362, 0.00000000],
    ]
)


def _encoder(**kwargs):
    return EcapaTdnnEncoder(
        input_size=20, model_scale=2, ndim=16, output_size=24, **kwargs
    )


def _dilations(encoder):
    blocks = [encoder.layer1, encoder.layer2, encoder.layer3, *encoder.extra_layers]
    return [block.convs[0].dilation[0] for block in blocks]


def _reference(encoder, x):
    """The forward pass written out: each block takes the stem plus all earlier."""
    h = encoder.bn(encoder.relu(encoder.conv(x.permute(0, 2, 1))))
    x1 = encoder.layer1(h)
    x2 = encoder.layer2(h + x1)
    x3 = encoder.layer3(h + x1 + x2)
    outputs = [x1, x2, x3]
    total = h + x1 + x2 + x3
    for block in encoder.extra_layers:
        outputs.append(block(total))
        total = total + outputs[-1]
    return encoder.relu(encoder.layer4(torch.cat(outputs, dim=1)))


def test_default_matches_the_encoder_before_this_change():
    torch.manual_seed(0)
    encoder = _encoder().eval()
    x = torch.randn(2, 50, 20)

    assert not any(key.startswith("extra_layers") for key in encoder.state_dict())
    assert encoder.layer4.weight.shape == (24, 3 * 16, 1)
    assert _dilations(encoder) == [2, 3, 4]
    with torch.no_grad():
        y = encoder(x)
        assert torch.equal(y, _reference(encoder, x))
    # Convolutions may differ in the last bits across CPUs.
    torch.testing.assert_close(y[0, :4, :4], BEFORE_THIS_CHANGE, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dilations", [[2, 3, 4, 5], [2, 3, 4, 2, 3, 4]])
def test_extra_blocks_follow_the_reference_forward(dilations):
    encoder = _encoder(dilations=dilations).eval()
    x = torch.randn(2, 50, 20)

    assert _dilations(encoder) == dilations
    assert len(encoder.extra_layers) == len(dilations) - 3
    assert encoder.layer4.in_channels == len(dilations) * 16
    with torch.no_grad():
        assert torch.equal(encoder(x), _reference(encoder, x))


def test_num_blocks_and_dilations_build_the_same_model():
    by_count = _encoder(num_blocks=4).eval()
    by_dilations = _encoder(dilations=[2, 3, 4, 5]).eval()
    by_dilations.load_state_dict(by_count.state_dict())
    x = torch.randn(2, 50, 20)

    assert _dilations(by_count) == _dilations(by_dilations) == [2, 3, 4, 5]
    with torch.no_grad():
        assert torch.equal(by_count(x), by_dilations(x))


def test_a_four_block_checkpoint_needs_four_blocks(tmp_path):
    trained = _encoder(num_blocks=4).eval()
    torch.save(trained.state_dict(), tmp_path / "encoder.pth")
    state = torch.load(tmp_path / "encoder.pth", weights_only=True)
    x = torch.randn(2, 50, 20)

    loaded = _encoder(num_blocks=4).eval()
    loaded.load_state_dict(state)
    with torch.no_grad():
        assert torch.equal(loaded(x), trained(x))
    with pytest.raises(RuntimeError, match="extra_layers|layer4"):
        _encoder().load_state_dict(state)


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


def test_rejects_unknown_arguments():
    with pytest.raises(TypeError):
        _encoder(num_block=4)
