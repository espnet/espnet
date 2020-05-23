import pytest

kenlm = pytest.importorskip("kenlm")
from math import isclose

test_sens = ["I like apple", "you love coffee"]


@pytest.mark.parameterize(test_sens)
def test_ngram_build(test_sens):
    lm = kenlm("test.arpa")
    assert isclose(lm.score(test_sens[0]), -1.04)
    assert isclose(lm.score(test_sens[1]), -1.18)
