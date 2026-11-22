import pytest
import torch

from espnet2.rst.decoder.dac_vocoder import (
    VOCODERS,
    DACVocoder,
    HiFiGANVocoder,
    build_vocoder,
)


def test_dac_decoder_shapes():
    vocoder = DACVocoder(input_dim=8, channels=64, rates=[2, 2])
    assert vocoder.upsample_factor == 4
    x = torch.randn(2, 8, 10)
    y = vocoder(x)
    assert y.shape == (2, 1, 40)
    assert y.abs().max() <= 1.0  # tanh output
    assert vocoder.generate(x.transpose(1, 2)).shape == (2, 40)


def test_dac_decoder_remove_weight_norm_keeps_output():
    vocoder = DACVocoder(input_dim=8, channels=64, rates=[2, 2]).eval()
    x = torch.randn(1, 8, 6)
    with torch.no_grad():
        before = vocoder(x)
        vocoder.remove_weight_norm()
        after = vocoder(x)
        vocoder.remove_weight_norm()  # idempotent
        again = vocoder(x)
    torch.testing.assert_close(before, after, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(after, again)
    assert not any(name.endswith("weight_g") for name, _ in vocoder.named_parameters())


def test_dac_decoder_ordered_parameters_cover_everything():
    vocoder = DACVocoder(input_dim=8, channels=64, rates=[2, 2])
    vocoder.remove_weight_norm()
    ordered = vocoder._ordered_parameters()
    assert len(ordered) == len(list(vocoder.parameters()))
    assert all(isinstance(p, torch.nn.Parameter) for _, p in ordered)


def test_hifigan_wrapper_shapes():
    vocoder = HiFiGANVocoder(
        input_dim=8,
        channels=16,
        upsample_scales=[2, 2],
        upsample_kernel_sizes=[4, 4],
        resblock_kernel_sizes=[3],
        resblock_dilations=[[1, 3]],
    )
    assert vocoder.upsample_factor == 4
    x = torch.randn(2, 8, 10)
    assert vocoder(x).shape == (2, 1, 40)
    assert vocoder.generate(x.transpose(1, 2)).shape == (2, 40)
    vocoder.remove_weight_norm()


def test_build_vocoder_registry():
    assert set(VOCODERS) == {"dac", "hifigan"}
    dac = build_vocoder("dac", 8, {"channels": 64, "rates": [2, 2]})
    assert isinstance(dac, DACVocoder)
    hifigan = build_vocoder(
        "hifigan",
        8,
        {
            "channels": 16,
            "upsample_scales": [2, 2],
            "upsample_kernel_sizes": [4, 4],
            "resblock_kernel_sizes": [3],
            "resblock_dilations": [[1, 3]],
        },
    )
    assert isinstance(hifigan, HiFiGANVocoder)
    with pytest.raises(ValueError):
        build_vocoder("unknown", 8)
