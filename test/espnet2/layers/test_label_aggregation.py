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
