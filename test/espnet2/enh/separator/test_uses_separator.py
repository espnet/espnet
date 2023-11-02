import pytest
import torch
from torch_complex import ComplexTensor

from espnet2.enh.separator.uses_separator import USESSeparator


@pytest.mark.parametrize("use_builtin_complex", [True, False])
@pytest.mark.parametrize("n_mics", [1, 3])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("ref_channel", [None, 0])
def test_uses_separator_forward_backward(
    use_builtin_complex, n_mics, num_spk, ref_channel
):
    model = USESSeparator(
        input_dim=None,
        num_spk=num_spk,
        enc_channels=12,
        bottleneck_size=8,
        num_blocks=3,
        num_spatial_blocks=2,
        ref_channel=ref_channel,
        segment_size=10,
        memory_size=2,
        # Transformer-related arguments
        hidden_size=4,
        att_heads=2,
        ch_mode="tac",
        ch_att_dim=8,
    )
    model.train()

    real = torch.rand(2, 18, n_mics, 33)
    imag = torch.rand(2, 18, n_mics, 33)
    if use_builtin_complex:
        x = torch.complex(real, imag)
    else:
        x = ComplexTensor(real, imag)
    x_lens = torch.tensor([18, 16], dtype=torch.long)

    output, flens, others = model(x, ilens=x_lens)
    assert len(output) == num_spk
    sum(output).abs().mean().backward()


@pytest.mark.parametrize("n_mics", [1, 3])
@pytest.mark.parametrize("num_spk", [1])
@pytest.mark.parametrize(
    "memory_types, mode",
    [(1, "no_dereverb"), (2, "dereverb"), (2, "both")],
)
@pytest.mark.parametrize("ch_mode", ["tac", "att"])
def test_uses_separator_mem_tokens(n_mics, num_spk, memory_types, mode, ch_mode):
    model = USESSeparator(
        input_dim=None,
        num_spk=num_spk,
        enc_channels=12,
        bottleneck_size=8,
        num_blocks=3,
        num_spatial_blocks=2,
        ref_channel=0,
        segment_size=10,
        memory_size=2,
        memory_types=memory_types,
        # Transformer-related arguments
        hidden_size=4,
        att_heads=2,
        ch_mode=ch_mode,
        ch_att_dim=8,
    )
    model.train()

    real = torch.rand(2, 18, n_mics, 33)
    imag = torch.rand(2, 18, n_mics, 33)
    x = ComplexTensor(real, imag)
    x_lens = torch.tensor([18, 16], dtype=torch.long)

    output, flens, others = model(x, ilens=x_lens, additional={"mode": mode})
    assert len(output) == num_spk
    sum(output).abs().mean().backward()


def test_uses_separator_invalid_type():
    with pytest.raises(AssertionError):
        USESSeparator(input_dim=None, ch_mode="fff")
