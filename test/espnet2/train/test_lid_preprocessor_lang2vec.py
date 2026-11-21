import sys

import numpy as np
import pytest

from espnet2.train.preprocessor import LIDPreprocessor


def make_preprocessor(tmp_path, contents):
    labels = tmp_path / "lang2utt"
    labels.write_text("eng example\ntok toki\n")
    table = tmp_path / "targets.txt"
    table.write_text(contents)
    return LIDPreprocessor(
        train=True,
        lang2utt=str(labels),
        lang2vec_table=str(table),
        fix_duration=False,
        noise_apply_prob=0.0,
        rir_apply_prob=0.0,
    )


def test_table_targets_and_missing_mask_without_lang2vec(tmp_path, monkeypatch):
    # Library code must work without importing the optional recipe package.
    monkeypatch.setitem(sys.modules, "lang2vec", None)
    preprocessor = make_preprocessor(tmp_path, "eng 0.1 0.2 0.3\n")
    for language, expected_mask in [("eng", 1), ("tok", 0)]:
        data = preprocessor(
            "utt",
            {
                "speech": np.ones(800, dtype=np.float32),
                "lid_labels": language,
            },
        )
        assert data["lid_labels"].item() == preprocessor.lang2label[language]
        assert data["lang2vecs"].shape == (3,)
        assert data["lang2vecs"].dtype == np.float32
        assert data["lang2vec_mask"].item() == expected_mask
        expected = [0.1, 0.2, 0.3] if expected_mask else [0, 0, 0]
        np.testing.assert_allclose(data["lang2vecs"], expected)


@pytest.mark.parametrize(
    "contents,match",
    [
        ("", "Empty"),
        ("eng\n", "Invalid"),
        ("eng nan\n", "Invalid"),
        ("eng 1\neng 2\n", "Duplicate"),
        ("eng 1 2\ntok 1\n", "Inconsistent"),
    ],
)
def test_invalid_tables_fail(tmp_path, contents, match):
    with pytest.raises(ValueError, match=match):
        make_preprocessor(tmp_path, contents)
