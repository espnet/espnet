import pytest
import torch

from espnet2.asr_transducer.encoder.blocks.conv1d import Conv1d
from espnet2.asr_transducer.encoder.modules.positional_encoding import (
    RelPositionalEncoding,
)


@pytest.mark.parametrize(
    "conv_conf",
    [
        {"kernel_size": 3},
        {"kernel_size": 2, "dilation": 2},
        {"kernel_size": 3, "stride": 2},
    ],
)
def test_conv1d_mask_matches_unpadded_outputs(conv_conf):
    block = Conv1d(4, 4, **conv_conf).eval()
    lengths = torch.tensor([19, 12, 7])
    x = torch.randn(3, int(lengths.max()), 4)
    mask = torch.arange(x.size(1))[None] >= lengths[:, None]
    pos_enc = RelPositionalEncoding(4, 0.0)(x)

    with torch.no_grad():
        out, out_mask, _ = block(x, pos_enc, mask)
        # Independently measure each convolution with its padding removed.
        expected = [
            block.conv(x[i : i + 1, :n].transpose(1, 2)).size(2)
            for i, n in enumerate(lengths)
        ]

    assert out_mask.shape == out.shape[:2]
    assert (~out_mask).sum(1).tolist() == expected
    assert torch.equal(
        out_mask, torch.arange(out.size(1))[None] >= torch.tensor(expected)[:, None]
    )
