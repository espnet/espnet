import pytest
import torch
from packaging.version import parse as V
from torch_complex import ComplexTensor

from espnet2.enh.separator.dccrn_separator import DCCRNSeparator

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


@pytest.mark.parametrize("input_dim", [9])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("rnn_layer", [2, 3])
@pytest.mark.parametrize("rnn_units", [256])
@pytest.mark.parametrize("masking_mode", ["E", "C", "R"])
@pytest.mark.parametrize("use_clstm", [True, False])
@pytest.mark.parametrize("bidirectional", [True, False])
@pytest.mark.parametrize("use_cbn", [True, False])
@pytest.mark.parametrize("kernel_size", [5])
@pytest.mark.parametrize("use_builtin_complex", [True, False])
@pytest.mark.parametrize("use_noise_mask", [True, False])
def test_dccrn_separator_forward_backward_complex(
    input_dim,
    num_spk,
    rnn_layer,
    rnn_units,
    masking_mode,
    use_clstm,
    bidirectional,
    use_cbn,
    kernel_size,
    use_builtin_complex,
    use_noise_mask,
):
    model = DCCRNSeparator(
        input_dim=input_dim,
        num_spk=num_spk,
        rnn_layer=rnn_layer,
        rnn_units=rnn_units,
        masking_mode=masking_mode,
        use_clstm=use_clstm,
        bidirectional=bidirectional,
        use_cbn=use_cbn,
        kernel_size=kernel_size,
        kernel_num=[32, 64, 128,],
        use_builtin_complex=use_builtin_complex,
        use_noise_mask=use_noise_mask,
    )
    model.train()

    real = torch.rand(2, 10, input_dim)
    imag = torch.rand(2, 10, input_dim)
    x = ComplexTensor(real, imag)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    masked, flens, others = model(x, ilens=x_lens)

    if use_builtin_complex and is_torch_1_9_plus:
        assert isinstance(masked[0], torch.Tensor)
    else:
        assert isinstance(masked[0], ComplexTensor)
    assert len(masked) == num_spk

    masked[0].abs().mean().backward()


def test_dccrn_separator_invalid_type():
    with pytest.raises(ValueError):
        DCCRNSeparator(
            input_dim=10, masking_mode="fff",
        )


def test_rnn_separator_output():
    real = torch.rand(2, 10, 9)
    imag = torch.rand(2, 10, 9)
    x = ComplexTensor(real, imag)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    for num_spk in range(1, 3):
        model = DCCRNSeparator(input_dim=9, num_spk=num_spk, kernel_num=[32, 64, 128,],)
        model.eval()
        specs, _, others = model(x, x_lens)
        assert isinstance(specs, list)
        assert isinstance(others, dict)
        for n in range(num_spk):
            assert "mask_spk{}".format(n + 1) in others
            assert specs[n].shape == others["mask_spk{}".format(n + 1)].shape
