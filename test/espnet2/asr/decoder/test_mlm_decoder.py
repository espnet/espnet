import pytest
import torch

from espnet2.asr.decoder.mlm_decoder import MLMDecoder


@pytest.mark.parametrize("input_layer", ["linear", "embed"])
@pytest.mark.parametrize("normalize_before", [True, False])
@pytest.mark.parametrize("use_output_layer", [True, False])
def test_MLMDecoder_backward(input_layer, normalize_before, use_output_layer):
    vocab_size = 10
    decoder = MLMDecoder(
        vocab_size,
        12,
        linear_units=10,
        num_blocks=2,
        input_layer=input_layer,
        normalize_before=normalize_before,
        use_output_layer=use_output_layer,
    )
    x = torch.randn(2, 9, 12)
    x_lens = torch.tensor([9, 7], dtype=torch.long)
    if input_layer == "embed":
        t = torch.randint(0, vocab_size + 1, [2, 4], dtype=torch.long)
    else:
        t = torch.randn(2, 4, vocab_size + 1)
    t_lens = torch.tensor([4, 3], dtype=torch.long)
    z_all, ys_in_lens = decoder(x, x_lens, t, t_lens)
    z_all.sum().backward()


def test_MLMDecoder_invalid_type():
    with pytest.raises(ValueError):
        MLMDecoder(10, 12, input_layer="foo")
