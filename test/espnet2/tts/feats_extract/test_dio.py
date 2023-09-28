import pytest
import torch

from espnet2.tts.feats_extract.dio import Dio


@pytest.mark.parametrize("use_continuous_f0", [False, True])
@pytest.mark.parametrize("use_log_f0", [False, True])
@pytest.mark.parametrize(
    "use_token_averaged_f0, reduction_factor", [(False, 1), (True, 1), (True, 3)]
)
def test_forward(
    use_continuous_f0, use_log_f0, use_token_averaged_f0, reduction_factor
):
    layer = Dio(
        n_fft=128,
        hop_length=64,
        f0min=40,
        f0max=800,
        fs="16k",
        use_continuous_f0=use_continuous_f0,
        use_log_f0=use_log_f0,
        use_token_averaged_f0=use_token_averaged_f0,
        reduction_factor=reduction_factor,
    )
    xs = torch.randn(2, 384)
    if not use_token_averaged_f0:
        layer(xs, torch.LongTensor([384, 128]))
    else:
        ds = torch.div(
            torch.LongTensor([[3, 3, 1], [3, 0, 0]]),
            reduction_factor,
            rounding_mode="trunc",
        )
        dlens = torch.LongTensor([3, 1])
        ps, _ = layer(
            xs, torch.LongTensor([384, 128]), durations=ds, durations_lengths=dlens
        )
        assert torch.isnan(ps).sum() == 0


@pytest.mark.parametrize(
    "use_token_averaged_f0, reduction_factor", [(False, 1), (True, 1), (True, 3)]
)
def test_output_size(use_token_averaged_f0, reduction_factor):
    layer = Dio(
        n_fft=4,
        hop_length=1,
        f0min=40,
        f0max=800,
        fs="16k",
        use_token_averaged_f0=use_token_averaged_f0,
        reduction_factor=reduction_factor,
    )
    print(layer.output_size())


@pytest.mark.parametrize(
    "use_token_averaged_f0, reduction_factor", [(False, 1), (True, 1), (True, 3)]
)
def test_get_parameters(use_token_averaged_f0, reduction_factor):
    layer = Dio(
        n_fft=4,
        hop_length=1,
        f0min=40,
        f0max=800,
        fs="16k",
        use_token_averaged_f0=use_token_averaged_f0,
        reduction_factor=reduction_factor,
    )
    print(layer.get_parameters())
