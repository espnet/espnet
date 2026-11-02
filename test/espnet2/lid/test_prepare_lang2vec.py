"""Recipe handles missing features before the generic preprocessor reads a table."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

PATH = (
    Path(__file__).resolve().parents[3] / "egs2/geolid/lid1/local/prepare_lang2vec.py"
)
spec = importlib.util.spec_from_file_location("prepare_lang2vec", PATH)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


def features(languages, feature):
    if languages[0] in ("tok", "missing"):
        raise ValueError("not available")
    return {languages[0]: [0.1, 0.2, 0.3]}


def test_explicit_missing_language_exclusion():
    targets = prepare.prepare_targets(["eng", "tok"], "geo", {"tok"}, features)
    assert set(targets) == {"eng"}
    assert targets["eng"].dtype == np.float32
    with pytest.raises(ValueError, match="Missing lang2vec features: missing"):
        prepare.prepare_targets(["eng", "tok", "missing"], "geo", {"tok"}, features)


def test_missing_language_is_not_silently_replaced():
    with pytest.raises(ValueError, match="Missing lang2vec features: tok"):
        prepare.prepare_targets(["eng", "tok"], "geo", set(), features)
    with pytest.raises(ValueError, match="At least one"):
        prepare.prepare_targets(["tok"], "geo", {"tok"}, features)


@pytest.mark.parametrize("vector", [[float("nan")], [], [[1.0]]])
def test_invalid_targets_fail(vector):
    with pytest.raises(ValueError, match="Invalid"):
        prepare.prepare_targets(
            ["eng"], "geo", set(), lambda langs, feature: {"eng": vector}
        )
