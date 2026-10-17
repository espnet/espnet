import pytest
import torch

from espnet2.layers.label_aggregation import LabelAggregate


@pytest.mark.parametrize(
    ("input_label", "expected_output"),
    [
        (torch.ones(10, 20000, 2), torch.ones(10, 157, 2)),
        (torch.zeros(10, 20000, 2), torch.zeros(10, 157, 2)),
    ],
)
def test_LabelAggregate(input_label, expected_output):
    label_aggregate = LabelAggregate(win_length=512, hop_length=128, center=True)
    aggregated_label, _ = label_aggregate.forward(input=input_label)
    assert torch.equal(aggregated_label, expected_output)


def test_LabelAggregate_repr():
    label_aggregate = LabelAggregate(win_length=512, hop_length=128, center=True)
    print(label_aggregate)


def test_LabelAggregate_no_center():
    win_length, hop_length = 512, 128
    input_label = torch.zeros(2, 2000, 2)
    input_label[0, :1000, 0] = 1.0
    input_label[1, 300:, 1] = 1.0
    ilens = torch.tensor([2000, 1500])

    label_aggregate = LabelAggregate(
        win_length=win_length, hop_length=hop_length, center=False
    )
    aggregated_label, olens = label_aggregate.forward(input=input_label, ilens=ilens)

    # Same framing as torch.stft(..., center=False)
    frames = input_label.unfold(1, win_length, hop_length)
    expected = (frames.sum(dim=-1) > win_length // 2).float()
    expected_olens = (ilens - win_length) // hop_length + 1
    for b, olen in enumerate(expected_olens):
        expected[b, olen:] = 0.0

    assert aggregated_label.shape == (2, 12, 2)
    assert torch.equal(olens, expected_olens)
    assert torch.equal(aggregated_label, expected)
