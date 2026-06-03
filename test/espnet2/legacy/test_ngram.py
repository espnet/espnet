import os
from math import isclose

import pytest

kenlm = pytest.importorskip("kenlm")


root = os.path.dirname(os.path.abspath(__file__))
test_sens = ["I like apple", "you love coffee"]


@pytest.mark.parametrize("test_sens", [test_sens])
def test_ngram_build(test_sens):
    lm = kenlm.LanguageModel(os.path.join(root, "test.arpa"))
    assert isclose(lm.score(test_sens[0]), -1.04, rel_tol=0.01)
    assert isclose(lm.score(test_sens[1]), -1.18, rel_tol=0.01)
