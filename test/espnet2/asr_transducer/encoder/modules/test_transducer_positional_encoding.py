import pytest
import torch

from espnet2.asr_transducer.encoder.modules.positional_encoding import (
    RelPositionalEncoding,
)


@pytest.mark.parametrize("low_dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("grow", [False, True])
@pytest.mark.parametrize("left_context", [0, 7, 32])
def test_cache_precision_after_dtype_changes(low_dtype, grow, left_context):
    layer = RelPositionalEncoding(8, max_len=16)
    low_length = 24 if grow else 8
    layer(torch.zeros(1, low_length, 8, dtype=low_dtype), left_context)
    cache_length = max(16, low_length + left_context)

    for dtype in [torch.float32, torch.float64]:
        x = torch.zeros(1, 8, 8, dtype=dtype)
        expected = RelPositionalEncoding(8, max_len=cache_length)(x, left_context)
        actual = layer(x, left_context)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert layer.pe.shape[1] == 2 * cache_length - 1
        assert actual.shape[1] == 2 * x.shape[1] + left_context - 1

        cache = layer.pe
        torch.testing.assert_close(layer(x, left_context), expected, rtol=0, atol=0)
        assert layer.pe is cache
