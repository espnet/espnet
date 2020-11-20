import pytest
import torch

from espnet2.tts.feats_extract.energy import Energy


@pytest.mark.parametrize(
    "use_token_averaged_f0, reduction_factor", [(False, 1), (True, 1), (True, 3)]
)
def test_forward(use_token_averaged_energy, reduction_factor):
    layer = Energy(
        n_fft=128,
        hop_length=64,
        fs="16k",
        use_token_averaged_energy=use_token_averaged_energy,
    )
    xs = torch.randn(2, 384)
    if not use_token_averaged_energy:
        es, elens = layer(xs, torch.LongTensor([384, 128]))
        assert es.shape[1] == max(elens)
    else:
        ds = torch.LongTensor([[3, 3, 1], [3, 0, 0]])
        dlens = torch.LongTensor([3, 1])
        es, _ = layer(
            xs, torch.LongTensor([384, 128]), durations=ds, durations_lengths=dlens
        )
        assert torch.isnan(es).sum() == 0


def test_output_size():
    layer = Energy(n_fft=4, hop_length=1, fs="16k")
    print(layer.output_size())


def test_get_parameters():
    layer = Energy(n_fft=4, hop_length=1, fs="16k")
    print(layer.get_parameters())
