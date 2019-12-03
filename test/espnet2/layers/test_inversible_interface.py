import pytest

from espnet2.layers.inversible_interface import InversibleInterface


class A(InversibleInterface):
    def inverse(self, input, input_lengths=None):
        super().inverse(input, input_lengths)


def test_abstract_method():
    a = A()
    with pytest.raises(NotImplementedError):
        a.inverse(1, 1)
