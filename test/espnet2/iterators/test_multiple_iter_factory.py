import pytest

from espnet2.iterators.abs_iter_factory import AbsIterFactory
from espnet2.iterators.multiple_iter_factory import MultipleIterFactory


class IterFactory(AbsIterFactory):
    def build_iter(self, epoch: int, shuffle: bool = None):
        return range(3)


@pytest.mark.parametrize("shuffle", [True, False])
def test_MultpleIterFactory(shuffle):
    iter_factory = MultipleIterFactory(
        build_funcs=[lambda: IterFactory(), lambda: IterFactory()], shuffle=shuffle,
    )
    assert [i for i in iter_factory.build_iter(0)] == [0, 1, 2, 0, 1, 2]
