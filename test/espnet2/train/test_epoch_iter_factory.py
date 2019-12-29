from espnet2.train.epoch_iter_factory import EpochIterFactory


class Dataset:
    def __getitem__(self, item):
        return item


def test_EpochIterFactory():
    dataset = Dataset()
    batches = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    iter_factory = EpochIterFactory(
        dataset=dataset, batches=batches, num_batches_per_epoch=3
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


def test_EpochIterFactory_deterministic():
    dataset = Dataset()
    batches = [[0, 1], [2, 3], [4, 5], [6, 7], [8, 9]]
    iter_factory = EpochIterFactory(
        dataset=dataset, batches=batches, num_batches_per_epoch=3, shuffle=True
    )

    for i in range(1, 10):
        for v, v2 in zip(iter_factory.build_iter(i), iter_factory.build_iter(i)):
            assert (v == v2).all()
