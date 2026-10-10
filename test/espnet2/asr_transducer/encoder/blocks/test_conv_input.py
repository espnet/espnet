import pytest
import torch

from espnet2.asr_transducer.encoder.blocks.conv_input import ConvInput


@pytest.mark.parametrize("three_dimensional_mask", [False, True])
@pytest.mark.parametrize(
    "vgg_like,factor", [(False, 2), (False, 4), (False, 6), (True, 4), (True, 6)]
)
def test_mask_shape_and_lengths(vgg_like, factor, three_dimensional_mask):
    """Both public mask shapes must describe the actual unpadded outputs."""
    block = ConvInput(
        input_size=80,
        conv_size=(4, 4) if vgg_like else 4,
        subsampling_factor=factor,
        vgg_like=vgg_like,
        output_size=8,
    ).eval()
    lengths = torch.tensor([64, 40, 24])
    speech = torch.randn(3, 64, 80)
    mask = torch.arange(64).unsqueeze(0) >= lengths.unsqueeze(1)
    speech[mask] = 0
    if three_dimensional_mask:
        mask = mask.unsqueeze(1)

    with torch.no_grad():
        output, output_mask = block(speech, mask)
        expected = [
            block(speech[i : i + 1, :length])[0].size(1)
            for i, length in enumerate(lengths)
        ]

    assert output_mask.shape == (*mask.shape[:-1], output.size(1))
    assert output_mask.dtype == torch.bool
    assert output_mask.eq(0).sum(-1).flatten().tolist() == expected
