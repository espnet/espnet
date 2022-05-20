import numpy as np

from espnet2.iterators.chunk_iter_factory import ChunkIterFactory
from espnet2.train.collate_fn import CommonCollateFn


class Dataset:
    def __init__(self):
        self.data = {
            "a": np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            "b": np.array([8, 9, 10, 11, 12]),
        }

    def __getitem__(self, item):
        return item, {"dummy": self.data["a"]}


def test_ChunkIterFactory():
    dataset = Dataset()
    collatefn = CommonCollateFn()
    batches = [["a"], ["b"]]
    iter_factory = ChunkIterFactory(
        dataset=dataset,
        batches=batches,
        batch_size=2,
        chunk_length=3,
        collate_fn=collatefn,
    )

    for key, batch in iter_factory.build_iter(0):
        for k, v in batch.items():
            assert v.shape == (2, 3)
