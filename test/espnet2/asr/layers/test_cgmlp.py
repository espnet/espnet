import torch

from espnet2.asr.layers.cgmlp import ConvolutionalSpatialGatingUnit
from espnet2.legacy.nets.pytorch_backend.nets_utils import make_pad_mask


def test_csgu_masks_before_the_conv():
    """The LayerNorm writes its bias into the padded frames; the mask is
    applied after it, right before the convolution reads them."""
    torch.manual_seed(0)
    unit = ConvolutionalSpatialGatingUnit(
        size=8,
        kernel_size=5,
        dropout_rate=0.0,
        use_linear_after_conv=False,
        gate_activation="identity",
    ).eval()
    # at init the convolution is near-identity (std 1e-6 weights) and the
    # LayerNorm bias is zero, so a padded frame stays exactly zero and nothing
    # leaks; a trained unit has neither property, so give it real parameters
    torch.nn.init.normal_(unit.conv.weight)
    torch.nn.init.normal_(unit.norm.bias)
    x = torch.randn(2, 10, 8)
    x[1, 6:] = 0.0
    mask = (~make_pad_mask(torch.tensor([10, 6]), maxlen=10)).unsqueeze(1)
    with torch.no_grad():
        ref = unit(x[1:, :6])
        unmasked = unit(x)
        masked = unit(x, mask_pad=mask)
    torch.testing.assert_close(masked[1, :6], ref[0], rtol=1e-6, atol=1e-6)
    assert (unmasked[1, :6] - ref[0]).abs().max() > 1e-4
