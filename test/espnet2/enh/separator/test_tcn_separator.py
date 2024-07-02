import pytest
import torch
from torch import Tensor
from torch_complex import ComplexTensor

from espnet2.enh.separator.tcn_separator import TCNSeparator


@pytest.mark.parametrize("input_dim", [5])
@pytest.mark.parametrize("bottleneck_dim", [5])
@pytest.mark.parametrize("hidden_dim", [5])
@pytest.mark.parametrize("kernel", [3])
@pytest.mark.parametrize("layer", [1, 3])
@pytest.mark.parametrize("stack", [1, 3])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("nonlinear", ["relu", "sigmoid", "tanh", "linear"])
@pytest.mark.parametrize("norm_type", ["BN", "gLN", "cLN"])
@pytest.mark.parametrize("masking", [True, False])
def test_tcn_separator_forward_backward_complex(
    input_dim,
    layer,
    num_spk,
    nonlinear,
    stack,
    bottleneck_dim,
    hidden_dim,
    kernel,
    causal,
    norm_type,
    masking,
):
    model = TCNSeparator(
        input_dim=input_dim,
        num_spk=num_spk,
        layer=layer,
        stack=stack,
        bottleneck_dim=bottleneck_dim,
        hidden_dim=hidden_dim,
        kernel=kernel,
        causal=causal,
        norm_type=norm_type,
        nonlinear=nonlinear,
        masking=masking,
    )
    model.train()

    real = torch.rand(2, 10, input_dim)
    imag = torch.rand(2, 10, input_dim)
    x = ComplexTensor(real, imag)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    masked, flens, others = model(x, ilens=x_lens)

    if masking:
        assert isinstance(masked[0], ComplexTensor)
    assert len(masked) == num_spk

    masked[0].abs().mean().backward()


@pytest.mark.parametrize("input_dim", [5])
@pytest.mark.parametrize("bottleneck_dim", [5])
@pytest.mark.parametrize("hidden_dim", [5])
@pytest.mark.parametrize("kernel", [3])
@pytest.mark.parametrize("layer", [1, 2])
@pytest.mark.parametrize("stack", [1, 2])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("nonlinear", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("norm_type", ["BN", "gLN", "cLN"])
@pytest.mark.parametrize("masking", [True, False])
def test_tcn_separator_forward_backward_real(
    input_dim,
    layer,
    num_spk,
    nonlinear,
    stack,
    bottleneck_dim,
    hidden_dim,
    kernel,
    causal,
    norm_type,
    masking,
):
    model = TCNSeparator(
        input_dim=input_dim,
        num_spk=num_spk,
        layer=layer,
        stack=stack,
        bottleneck_dim=bottleneck_dim,
        hidden_dim=hidden_dim,
        kernel=kernel,
        causal=causal,
        norm_type=norm_type,
        nonlinear=nonlinear,
        masking=masking,
    )

    x = torch.rand(2, 10, input_dim)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    masked, flens, others = model(x, ilens=x_lens)

    assert isinstance(masked[0], Tensor)
    assert len(masked) == num_spk

    masked[0].abs().mean().backward()


def test_tcn_separator_invalid_type():
    with pytest.raises(ValueError):
        TCNSeparator(
            input_dim=10,
            nonlinear="fff",
        )
    with pytest.raises(ValueError):
        TCNSeparator(
            input_dim=10,
            norm_type="xxx",
        )


def test_tcn_separator_output():
    x = torch.rand(2, 10, 10)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    for num_spk in range(1, 3):
        model = TCNSeparator(
            input_dim=10,
            layer=num_spk,
            stack=2,
            bottleneck_dim=3,
            hidden_dim=3,
            kernel=3,
            causal=False,
        )
        model.eval()
        specs, _, others = model(x, x_lens)
        assert isinstance(specs, list)
        assert isinstance(others, dict)
        for n in range(num_spk):
            assert "mask_spk{}".format(n + 1) in others
            assert specs[n].shape == others["mask_spk{}".format(n + 1)].shape


def test_tcn_streaming():
    SEQ_LEN = 100
    num_spk = 2
    BS = 2
    separator = TCNSeparator(
        input_dim=128,
        num_spk=2,
        layer=2,
        stack=3,
        bottleneck_dim=32,
        hidden_dim=64,
        kernel=3,
        causal=True,
        norm_type="cLN",
    )
    separator.eval()
    input_feature = torch.randn((BS, SEQ_LEN, 128))
    ilens = torch.LongTensor([SEQ_LEN] * BS)
    with torch.no_grad():
        seq_output, _, _ = separator.forward(input_feature, ilens=ilens)

        state = None
        stream_outputs = []
        for i in range(SEQ_LEN):
            frame = input_feature[:, i : i + 1, :]
            frame_out, state, _ = separator.forward_streaming(frame, state)
            stream_outputs.append(frame_out)
        for i in range(SEQ_LEN):
            for s in range(num_spk):
                torch.testing.assert_close(
                    stream_outputs[i][s], seq_output[s][:, i : i + 1, :]
                )
