import pytest
import torch

from espnet2.tts.feats_extract.energy import Energy


@pytest.mark.parametrize(
    "use_token_averaged_energy, reduction_factor", [(False, None), (True, 1), (True, 3)]
)
def test_forward(use_token_averaged_energy, reduction_factor):
    layer = Energy(
        n_fft=128,
        hop_length=64,
        fs="16k",
        use_token_averaged_energy=use_token_averaged_energy,
        reduction_factor=reduction_factor,
    )
    xs = torch.randn(2, 384)
    if not use_token_averaged_energy:
        es, elens = layer(xs, torch.LongTensor([384, 128]))
        assert es.shape[1] == max(elens)
    else:
        ds = torch.div(
            torch.LongTensor([[3, 3, 1], [3, 0, 0]]),
            reduction_factor,
            rounding_mode="trunc",
        )
        dlens = torch.LongTensor([3, 1])
        es, _ = layer(
            xs, torch.LongTensor([384, 128]), durations=ds, durations_lengths=dlens
        )
        assert torch.isnan(es).sum() == 0


@pytest.mark.parametrize(
    "use_token_averaged_energy, reduction_factor", [(False, None), (True, 1), (True, 3)]
)
def test_output_size(use_token_averaged_energy, reduction_factor):
    layer = Energy(
        n_fft=4,
        hop_length=1,
        fs="16k",
        use_token_averaged_energy=use_token_averaged_energy,
        reduction_factor=reduction_factor,
    )
    print(layer.output_size())


@pytest.mark.parametrize(
    "use_token_averaged_energy, reduction_factor", [(False, None), (True, 1), (True, 3)]
)
def test_get_parameters(use_token_averaged_energy, reduction_factor):
    layer = Energy(
        n_fft=4,
        hop_length=1,
        fs="16k",
        use_token_averaged_energy=use_token_averaged_energy,
        reduction_factor=reduction_factor,
    )
    print(layer.get_parameters())
