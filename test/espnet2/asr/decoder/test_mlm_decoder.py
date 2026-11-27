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


def test_MLMDecoder_use_sdpa_matches_the_default_with_padding():
    vocab_size = 10
    x = torch.randn(2, 9, 12)
    x_lens = torch.tensor([9, 7], dtype=torch.long)
    t = torch.randint(0, vocab_size + 1, [2, 4], dtype=torch.long)
    # The padded target rows can attend to nothing in the self-attention.
    t_lens = torch.tensor([4, 2], dtype=torch.long)

    def build(**kwargs):
        torch.manual_seed(0)
        return MLMDecoder(vocab_size, 12, linear_units=10, num_blocks=2, **kwargs)

    sdpa_decoder = build(use_sdpa=True).eval()
    attentions = [m for m in sdpa_decoder.modules() if hasattr(m, "use_sdpa")]
    assert attentions and all(m.use_sdpa for m in attentions)

    default_out, _ = build().eval()(x, x_lens, t, t_lens)
    sdpa_out, _ = sdpa_decoder(x, x_lens, t, t_lens)
    assert torch.isfinite(sdpa_out).all()
    assert torch.allclose(default_out, sdpa_out, atol=1e-5)

    sdpa_out.sum().backward()
    for p in sdpa_decoder.parameters():
        assert p.grad is None or torch.isfinite(p.grad).all()
