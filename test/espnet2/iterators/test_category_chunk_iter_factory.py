import re

import numpy as np
import pytest

from espnet2.iterators.category_chunk_iter_factory import CategoryChunkIterFactory
from espnet2.train.collate_fn import CommonCollateFn


class Dataset:
    def __init__(self):
        self.data = {
            "a": np.array([0, 1, 2, 3, 4, 5, 6, 7]),
            "b": np.array([8, 9, 10, 11, 12]),
        }

    def __getitem__(self, item):
        return item, {"dummy": self.data["a"]}


@pytest.mark.parametrize("chunk_length", [3, "2,4", "3-5"])
def test_CategoryChunkIterFactory(chunk_length):
    dataset = Dataset()
    collatefn = CommonCollateFn()
    batches = [["a", "b"]]
    iter_factory = CategoryChunkIterFactory(
        dataset=dataset,
        batches=batches,
        batch_size=2,
        chunk_length=chunk_length,
        collate_fn=collatefn,
    )

    for key, batch in iter_factory.build_iter(0):
        for k, v in batch.items():
            if chunk_length == 3:
                assert v.shape == (2, 3)
            elif chunk_length == "2,4":
                assert v.ndim == 2 and v.shape[0] == 2 and v.shape[1] in (2, 4)
            elif chunk_length == "3-5":
                assert v.ndim == 2 and v.shape[0] == 2 and v.shape[1] in (3, 4, 5)


class Dataset2:
    def __init__(self):
        self.data = {
            "a": {
                "label": np.array([0, 1, 2, 3, 4, 5, 6, 7]),
                "aux": np.array([0, -1, -2, -3]),
            },
            "b": {"label": np.array([8, 9, 10, 11, 12]), "aux": np.array([-1, -2])},
        }

    def __getitem__(self, item):
        return item, self.data[item]


def test_CategoryChunkIterFactory_partial_chunking():
    dataset = Dataset2()
    collatefn = CommonCollateFn()
    batches = [["a", "b"]]
    iter_factory = CategoryChunkIterFactory(
        dataset=dataset,
        batches=batches,
        batch_size=2,
        chunk_length=3,
        collate_fn=collatefn,
        excluded_key_prefixes=["aux"],
    )

    for key, batch in iter_factory.build_iter(0):
        for k, v in batch.items():
            if iter_factory.excluded_key_pattern is not None and re.fullmatch(
                iter_factory.excluded_key_pattern, k
            ):
                assert v.shape in ((2, 2), (2, 4))
            else:
                assert v.shape == (2, 3)
