import sys
from types import ModuleType

import numpy as np

from espnet2.train.preprocessor import LIDPreprocessor


def test_vectors_are_float32(tmp_path, monkeypatch):
    package = ModuleType("lang2vec")
    module = ModuleType("lang2vec.lang2vec")
    module.get_features = lambda langs, feature: {langs[0]: [0.1] * 299}
    package.lang2vec = module
    monkeypatch.setitem(sys.modules, "lang2vec", package)
    monkeypatch.setitem(sys.modules, "lang2vec.lang2vec", module)
    labels = tmp_path / "lang2utt"
    labels.write_text("eng example\n")
    preprocessor = LIDPreprocessor(
        train=False, lang2utt=str(labels), use_lang2vec=True, lang2vec_type="geo"
    )
    assert preprocessor.lang2vec["eng"].dtype == np.float32


def test_missing_languages_raise(tmp_path, monkeypatch):
    import pytest

    package = ModuleType("lang2vec")
    module = ModuleType("lang2vec.lang2vec")

    def missing(langs, feature):
        raise ValueError("not available")

    module.get_features = missing
    package.lang2vec = module
    monkeypatch.setitem(sys.modules, "lang2vec", package)
    monkeypatch.setitem(sys.modules, "lang2vec.lang2vec", module)
    labels = tmp_path / "lang2utt"
    labels.write_text("eng example\nfra autre\n")
    with pytest.raises(ValueError, match="Missing lang2vec features: eng.*fra"):
        LIDPreprocessor(
            train=False, lang2utt=str(labels), use_lang2vec=True, lang2vec_type="geo"
        )
