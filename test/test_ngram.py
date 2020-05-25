import pytest
import os

from math import isclose

kenlm = pytest.importorskip("kenlm")

test_sens = ["I like apple", "you love coffee"]


@pytest.mark.parametrize("test_sens", [test_sens])
def test_ngram_build(test_sens):
    lm = kenlm.LanguageModel("test.arpa")
    assert isclose(lm.score(test_sens[0]), -1.04)
    assert isclose(lm.score(test_sens[1]), -1.18)
