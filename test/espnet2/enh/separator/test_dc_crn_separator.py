import pytest
import torch
from packaging.version import parse as V
from torch_complex import ComplexTensor

from espnet2.enh.layers.complex_utils import is_complex
from espnet2.enh.separator.dc_crn_separator import DC_CRNSeparator

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


@pytest.mark.parametrize("input_dim", [33, 65])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("input_channels", [[2, 4], [2, 4, 4]])
@pytest.mark.parametrize("enc_hid_channels", [2, 5])
@pytest.mark.parametrize("enc_layers", [2])
@pytest.mark.parametrize("glstm_groups", [2])
@pytest.mark.parametrize("glstm_layers", [1, 2])
@pytest.mark.parametrize("glstm_bidirectional", [True, False])
@pytest.mark.parametrize("glstm_rearrange", [True, False])
@pytest.mark.parametrize("mode", ["mapping", "masking"])
def test_dc_crn_separator_forward_backward_complex(
    input_dim,
    num_spk,
    input_channels,
    enc_hid_channels,
    enc_layers,
    glstm_groups,
    glstm_layers,
    glstm_bidirectional,
    glstm_rearrange,
    mode,
):
    model = DC_CRNSeparator(
        input_dim=input_dim,
        num_spk=num_spk,
        input_channels=input_channels,
        enc_hid_channels=enc_hid_channels,
        enc_kernel_size=(1, 3),
        enc_padding=(0, 1),
        enc_last_kernel_size=(1, 3),
        enc_last_stride=(1, 2),
        enc_last_padding=(0, 1),
        enc_layers=enc_layers,
        skip_last_kernel_size=(1, 3),
        skip_last_stride=(1, 1),
        skip_last_padding=(0, 1),
        glstm_groups=glstm_groups,
        glstm_layers=glstm_layers,
        glstm_bidirectional=glstm_bidirectional,
        glstm_rearrange=glstm_rearrange,
        mode=mode,
    )
    model.train()

    real = torch.rand(2, 10, input_dim)
    imag = torch.rand(2, 10, input_dim)
    x = torch.complex(real, imag) if is_torch_1_9_plus else ComplexTensor(real, imag)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    masked, flens, others = model(x, ilens=x_lens)

    assert is_complex(masked[0])
    assert len(masked) == num_spk

    masked[0].abs().mean().backward()


@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("input_channels", [[4, 4], [6, 4, 4]])
@pytest.mark.parametrize(
    "enc_kernel_size, enc_padding", [((1, 3), (0, 1)), ((1, 5), (0, 2))]
)
@pytest.mark.parametrize("enc_last_stride", [(1, 2)])
@pytest.mark.parametrize(
    "enc_last_kernel_size, enc_last_padding", [((1, 4), (0, 1)), ((1, 5), (0, 2))],
)
@pytest.mark.parametrize("skip_last_stride", [(1, 1)])
@pytest.mark.parametrize(
    "skip_last_kernel_size, skip_last_padding", [((1, 3), (0, 1)), ((1, 5), (0, 2))],
)
def test_dc_crn_separator_multich_input(
    num_spk,
    input_channels,
    enc_kernel_size,
    enc_padding,
    enc_last_kernel_size,
    enc_last_stride,
    enc_last_padding,
    skip_last_kernel_size,
    skip_last_stride,
    skip_last_padding,
):
    model = DC_CRNSeparator(
        input_dim=33,
        num_spk=num_spk,
        input_channels=input_channels,
        enc_hid_channels=2,
        enc_kernel_size=enc_kernel_size,
        enc_padding=enc_padding,
        enc_last_kernel_size=enc_last_kernel_size,
        enc_last_stride=enc_last_stride,
        enc_last_padding=enc_last_padding,
        enc_layers=3,
        skip_last_kernel_size=skip_last_kernel_size,
        skip_last_stride=skip_last_stride,
        skip_last_padding=skip_last_padding,
    )
    model.train()

    real = torch.rand(2, 10, input_channels[0] // 2, 33)
    imag = torch.rand(2, 10, input_channels[0] // 2, 33)
    x = torch.complex(real, imag) if is_torch_1_9_plus else ComplexTensor(real, imag)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    masked, flens, others = model(x, ilens=x_lens)

    assert is_complex(masked[0])
    assert len(masked) == num_spk

    masked[0].abs().mean().backward()


def test_dc_crn_separator_invalid_enc_layer():
    with pytest.raises(AssertionError):
        DC_CRNSeparator(
            input_dim=17, input_channels=[2, 2, 4], enc_layers=1,
        )


def test_dc_crn_separator_invalid_type():
    with pytest.raises(ValueError):
        DC_CRNSeparator(
            input_dim=17, input_channels=[2, 2, 4], mode="xxx",
        )


def test_dc_crn_separator_output():
    real = torch.rand(2, 10, 17)
    imag = torch.rand(2, 10, 17)
    x = torch.complex(real, imag) if is_torch_1_9_plus else ComplexTensor(real, imag)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    for num_spk in range(1, 3):
        model = DC_CRNSeparator(
            input_dim=17, num_spk=num_spk, input_channels=[2, 2, 4],
        )
        model.eval()
        specs, _, others = model(x, x_lens)
        assert isinstance(specs, list)
        assert isinstance(others, dict)
        for n in range(num_spk):
            assert "mask_spk{}".format(n + 1) in others
            assert specs[n].shape == others["mask_spk{}".format(n + 1)].shape
