import pytest
import torch

from espnet2.lid.frontend.s3prl_condition import S3prlFrontendCondition


@pytest.mark.parametrize(
    "layer,multilayer", [(0, False), ([0, 2], True), (None, False)]
)
def test_layer_selection_and_tiling(tiny_geolid, layer, multilayer):
    _, _, base = tiny_geolid
    frontend = S3prlFrontendCondition(
        frontend_conf={"path_or_url": str(base), "tile_factor": 2},
        layer=layer,
        multilayer_feature=multilayer,
    ).eval()
    with torch.no_grad():
        features, lengths, predictions = frontend.forward_with_condition(
            torch.randn(2, 880), torch.tensor([880, 810])
        )
    assert features.shape == (2, 880, 8)
    torch.testing.assert_close(lengths, torch.tensor([880, 810]))
    assert predictions == ()


def test_short_waveforms(tiny_geolid):
    _, model, _ = tiny_geolid
    model.eval()
    with torch.no_grad():
        features, lengths = model.frontend(torch.randn(2, 40), torch.tensor([40, 30]))
    assert features.shape[:2] == (2, 20)
    torch.testing.assert_close(lengths, torch.tensor([20, 15]))
