import pytest
import torch
from torch import Tensor
from torch_complex import ComplexTensor

from espnet2.enh.separator.skim_separator import SkiMSeparator


@pytest.mark.parametrize("input_dim", [5])
@pytest.mark.parametrize("layer", [1, 3])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("unit", [8])
@pytest.mark.parametrize("dropout", [0.0, 0.2])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("nonlinear", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("mem_type", ["hc", "c", "h", None])
@pytest.mark.parametrize("segment_size", [2, 4])
@pytest.mark.parametrize("seg_overlap", [False, True])
def test_skim_separator_forward_backward_complex(
    input_dim,
    layer,
    causal,
    unit,
    dropout,
    num_spk,
    nonlinear,
    mem_type,
    segment_size,
    seg_overlap,
):
    model = SkiMSeparator(
        input_dim=input_dim,
        causal=causal,
        num_spk=num_spk,
        nonlinear=nonlinear,
        layer=layer,
        unit=unit,
        segment_size=segment_size,
        dropout=dropout,
        mem_type=mem_type,
        seg_overlap=seg_overlap,
    )
    model.train()

    real = torch.rand(2, 10, input_dim)
    imag = torch.rand(2, 10, input_dim)
    x = ComplexTensor(real, imag)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    masked, flens, others = model(x, ilens=x_lens)

    assert isinstance(masked[0], ComplexTensor)
    assert len(masked) == num_spk

    masked[0].abs().mean().backward()


@pytest.mark.parametrize("input_dim", [5])
@pytest.mark.parametrize("layer", [1, 3])
@pytest.mark.parametrize("causal", [True, False])
@pytest.mark.parametrize("unit", [8])
@pytest.mark.parametrize("dropout", [0.0, 0.2])
@pytest.mark.parametrize("num_spk", [1, 2])
@pytest.mark.parametrize("nonlinear", ["relu", "sigmoid", "tanh"])
@pytest.mark.parametrize("mem_type", ["hc", "c", "h", "id", None])
@pytest.mark.parametrize("segment_size", [2, 4])
@pytest.mark.parametrize("seg_overlap", [False, True])
def test_skim_separator_forward_backward_real(
    input_dim,
    layer,
    causal,
    unit,
    dropout,
    num_spk,
    nonlinear,
    mem_type,
    segment_size,
    seg_overlap,
):
    model = SkiMSeparator(
        input_dim=input_dim,
        causal=causal,
        num_spk=num_spk,
        nonlinear=nonlinear,
        layer=layer,
        unit=unit,
        segment_size=segment_size,
        dropout=dropout,
        mem_type=mem_type,
        seg_overlap=seg_overlap,
    )
    model.train()

    x = torch.rand(2, 10, input_dim)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    masked, flens, others = model(x, ilens=x_lens)

    assert isinstance(masked[0], Tensor)
    assert len(masked) == num_spk

    masked[0].abs().mean().backward()


def test_skim_separator_invalid_type():
    with pytest.raises(ValueError):
        SkiMSeparator(
            input_dim=10,
            layer=2,
            unit=10,
            dropout=0.1,
            num_spk=2,
            nonlinear="fff",
            mem_type="aaa",
            segment_size=2,
        )


def test_skim_separator_output():
    x = torch.rand(2, 10, 10)
    x_lens = torch.tensor([10, 8], dtype=torch.long)

    for num_spk in range(1, 3):
        model = SkiMSeparator(
            input_dim=10,
            layer=2,
            unit=10,
            dropout=0.1,
            num_spk=2,
            nonlinear="relu",
            segment_size=2,
        )
        model.eval()
        specs, _, others = model(x, x_lens)
        assert isinstance(specs, list)
        assert isinstance(others, dict)
        assert x.shape == specs[0].shape
        for n in range(num_spk):
            assert "mask_spk{}".format(n + 1) in others
            assert specs[n].shape == others["mask_spk{}".format(n + 1)].shape


def test_skim_streaming():
    SEQ_LEN = 100
    num_spk = 2
    BS = 2
    separator = SkiMSeparator(
        input_dim=128,
        causal=True,
        num_spk=2,
        layer=2,
        unit=32,
        segment_size=100,
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
                torch.testing.assert_allclose(
                    stream_outputs[i][s], seq_output[s][:, i : i + 1, :]
                )
