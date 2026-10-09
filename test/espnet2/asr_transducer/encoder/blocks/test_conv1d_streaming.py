import pytest
import torch
from torch.nn import functional as F

from espnet2.asr_transducer.encoder.blocks.conv1d import Conv1d
from espnet2.asr_transducer.encoder.modules.positional_encoding import (
    RelPositionalEncoding,
)


@pytest.mark.parametrize("kernel_size,dilation", [(1, 1), (3, 1), (3, 2), (5, 3)])
def test_causal_conv1d_streaming_matches_full_sequence(kernel_size, dilation):
    block = Conv1d(4, 8, kernel_size, dilation=dilation, causal=True, relu=False)
    block.eval()
    x = torch.randn(1, 20, 4)
    mask = torch.zeros(1, x.size(1), dtype=torch.bool)
    positional_encoding = RelPositionalEncoding(4)
    context = dilation * (kernel_size - 1)

    with torch.no_grad():
        expected = block.conv(F.pad(x.transpose(1, 2), (context, 0))).transpose(1, 2)
        full, full_mask, _ = block(x, positional_encoding(x), mask)
        torch.testing.assert_close(full, expected)
        assert full.shape[:2] == full_mask.shape

        # Include chunks shorter than the effective convolution context, then
        # reset and encode again to check that no previous utterance leaks in.
        for _ in range(2):
            block.reset_streaming_cache(0, x.device)
            chunks = []
            offset = 0
            for size in [1, 2, 6, 1, 10]:
                chunk = x[:, offset : offset + size]
                out, _ = block.chunk_forward(
                    chunk, positional_encoding(chunk), mask[:, :size]
                )
                assert out.size(1) == size
                assert block.cache.size(2) == context
                chunks.append(out)
                offset += size
            torch.testing.assert_close(torch.cat(chunks, dim=1), expected)
