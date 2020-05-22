import pytest
import os
kenlm = pytest.importorskip("kenlm")
from math import isclose

from espnet.nets.pytorch_backend.lm.ngram import NgramFullScorer
from espnet.nets.pytorch_backend.lm.ngram import NgramPartScorer

@pytest.mark.parameterize()
def test_ngram_build():
    test_sens=["I like apple", "you love coffee"]
    lm = kenlm("test.arpa")
    assert isclose(lm.score(test_sens[0]), -1.04 )
    assert isclose(lm.score(test_sens[1]), -1.18 )
