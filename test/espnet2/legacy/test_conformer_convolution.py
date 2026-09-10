import torch

from espnet2.legacy.nets.pytorch_backend.conformer.convolution import (
    ConvolutionModule,
    mask_padded_frames,
)
from espnet2.legacy.nets.pytorch_backend.nets_utils import make_pad_mask


def test_mask_padded_frames_zeroes_only_the_padding():
    x = torch.randn(2, 3, 5)
    mask = (~make_pad_mask(torch.tensor([5, 3]), maxlen=5)).unsqueeze(1)
    out = mask_padded_frames(x, mask)
    assert torch.equal(out[0], x[0])
    assert torch.equal(out[1, :, :3], x[1, :, :3])
    assert out[1, :, 3:].abs().sum() == 0
    # a mask that cannot apply (another time axis, or None) is ignored
    assert mask_padded_frames(x, None) is x
    assert mask_padded_frames(x, mask[:, :, :1]) is x


def test_convolution_module_masks_before_the_depthwise_conv():
    """Masking the module input would not do: the pointwise conv and the GLU
    write their biases into the padded frames first."""
    torch.manual_seed(0)
    module = ConvolutionModule(8, kernel_size=5).eval()
    x = torch.randn(2, 10, 8)
    x[1, 6:] = 0.0
    mask = (~make_pad_mask(torch.tensor([10, 6]), maxlen=10)).unsqueeze(1)
    with torch.no_grad():
        ref = module(x[1:, :6])
        unmasked = module(x)
        masked = module(x, mask_pad=mask)
    torch.testing.assert_close(masked[1, :6], ref[0], rtol=1e-6, atol=1e-6)
    assert (unmasked[1, :6] - ref[0]).abs().max() > 1e-4
    # without padding the mask changes nothing
    torch.testing.assert_close(masked[0], unmasked[0], rtol=0, atol=0)
