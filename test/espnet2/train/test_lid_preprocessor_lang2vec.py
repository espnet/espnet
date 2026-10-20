import sys
from types import ModuleType

import numpy as np
import pytest

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


@pytest.mark.parametrize("other_missing", [False, True])
def test_explicit_exclusion_retains_labels(tmp_path, monkeypatch, other_missing):
    package = ModuleType("lang2vec")
    module = ModuleType("lang2vec.lang2vec")

    def features(langs, feature):
        assert langs != ["tok"]
        if langs == ["zzz"]:
            raise ValueError("not available")
        return {langs[0]: [0.1] * 299}

    module.get_features = features
    package.lang2vec = module
    monkeypatch.setitem(sys.modules, "lang2vec", package)
    monkeypatch.setitem(sys.modules, "lang2vec.lang2vec", module)
    labels = tmp_path / "lang2utt"
    labels.write_text(
        "eng example\ntok toki\n" + ("zzz absent\n" if other_missing else "")
    )
    kwargs = dict(
        train=True,
        lang2utt=str(labels),
        use_lang2vec=True,
        lang2vec_type="geo",
        lang2vec_exclude_languages=["tok"],
        fix_duration=False,
        noise_apply_prob=0.0,
        rir_apply_prob=0.0,
    )
    if other_missing:
        with pytest.raises(ValueError, match="Missing lang2vec features: zzz"):
            LIDPreprocessor(**kwargs)
        return
    preprocessor = LIDPreprocessor(**kwargs)
    for language, expected_mask in [("eng", 1), ("tok", 0)]:
        data = preprocessor(
            "utterance",
            {
                "speech": np.ones(800, dtype=np.float32),
                "lid_labels": language,
            },
        )
        assert data["lid_labels"].item() == preprocessor.lang2label[language]
        assert data["lang2vecs"].shape == (299,)
        assert data["lang2vec_mask"].item() == expected_mask


def test_excluding_every_language_raises(tmp_path):
    pytest.importorskip("lang2vec")
    labels = tmp_path / "lang2utt"
    labels.write_text("tok example\n")
    with pytest.raises(ValueError, match="At least one language"):
        LIDPreprocessor(
            train=False,
            lang2utt=str(labels),
            use_lang2vec=True,
            lang2vec_type="geo",
            lang2vec_exclude_languages=["tok"],
        )
