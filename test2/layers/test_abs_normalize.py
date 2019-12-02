import pytest

from espnet2.layers.abs_normalize import AbsNormalize


class A(AbsNormalize):
    def forward(self, input, input_lengths):
        super().forward(input, input_lengths)


def test_abstract_method():
    a = A()
    with pytest.raises(NotImplementedError):
        a.forward(1, 1)

