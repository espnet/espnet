import pytest
import torch

from espnet2.asr.decoder.transformer_decoder import TransformerDecoder


@pytest.mark.parametrize("input_layer", ["linear", "embed"])
@pytest.mark.parametrize("normalize_before", [True, False])
@pytest.mark.parametrize("use_output_layer", [True, False])
def test_TransformerDecoder_backward(input_layer, normalize_before, use_output_layer):
    decoder = TransformerDecoder(
        10,
        12,
        input_layer=input_layer,
        normalize_before=normalize_before,
        use_output_layer=use_output_layer,
    )
    x = torch.randn(2, 9, 12)
    x_lens = torch.tensor([9, 7], dtype=torch.long)
    if input_layer == "embed":
        t = torch.randint(0, 10, [2, 4], dtype=torch.long)
    else:
        t = torch.randn(2, 4, 10)
    t_lens = torch.tensor([4, 3], dtype=torch.long)
    z_all, ys_in_lens = decoder(x, x_lens, t, t_lens)
    z_all.sum().backward()


def test_TransformerDecoder_init_state():
    decoder = TransformerDecoder(10, 12)
    x = torch.randn(9, 12)
    state = decoder.init_state(x)
    t = torch.randint(0, 10, [4], dtype=torch.long)
    decoder.score(t, state, x)


def test_TransformerDecoder_invalid_type():
    with pytest.raises(ValueError):
        TransformerDecoder(10, 12, input_layer="foo")
