import pytest
import torch

from espnet2.iterators.sequence_iter_factory import SequenceIterFactory


class Dataset:
    def __getitem__(self, item):
        return item


def collate_func(x):
    return torch.tensor(x)


@pytest.mark.parametrize("collate", [None, collate_func])
def test_SequenceIterFactory_larger_than_num_iters(collate):
    dataset = Dataset()
    batches = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    iter_factory = SequenceIterFactory(
        dataset=dataset, batches=batches, num_iters_per_epoch=3, collate_fn=collate
    )

    seq = [
        [list(map(int, it)) for it in iter_factory.build_iter(i)] for i in range(1, 5)
    ]
    assert seq == [
        [[0, 1], [2, 3], [4, 5]],
        [[6, 7], [8, 9], [0, 1]],
        [[2, 3], [4, 5], [6, 7]],
        [[8, 9], [0, 1], [2, 3]],
    ]


@pytest.mark.parametrize("collate", [None, collate_func])
def test_SequenceIterFactory_smaller_than_num_iters(collate):
    dataset = Dataset()
    batches = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    iter_factory = SequenceIterFactory(
        dataset=dataset, batches=batches, num_iters_per_epoch=9, collate_fn=collate
    )

    seq = [
        [list(map(int, it)) for it in iter_factory.build_iter(i)] for i in range(1, 5)
    ]
    assert seq == [
        [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [0, 1], [2, 3], [4, 5], [6, 7]],
        [[8, 9], [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [0, 1], [2, 3], [4, 5]],
        [[6, 7], [8, 9], [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [0, 1], [2, 3]],
        [[4, 5], [6, 7], [8, 9], [0, 1], [2, 3], [4, 5], [6, 7], [8, 9], [0, 1]],
    ]


@pytest.mark.parametrize("collate", [None, collate_func])
@pytest.mark.parametrize("num_iters_per_epoch", [None, 3, 9])
def test_SequenceIterFactory_deterministic(collate, num_iters_per_epoch):
    dataset = Dataset()
    batches = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    iter_factory = SequenceIterFactory(
        dataset=dataset,
        batches=batches,
        num_iters_per_epoch=3,
        shuffle=True,
        collate_fn=collate,
    )

    for i in range(1, 10):
        for v, v2 in zip(iter_factory.build_iter(i), iter_factory.build_iter(i)):
            assert (v == v2).all()
