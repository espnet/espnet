"""Verify VOiCES sample loading and dataset selectors."""

import numpy as np
import pytest
import soundfile as sf

from egs3.voices.asr.dataset import Dataset, DatasetBuilder
from egs3.voices.asr.dataset.builder import SPLITS


def test_waveforms_and_condition_selection(corpus):
    """Read original float32 samples and expose only model input fields."""
    recipe, _ = corpus
    DatasetBuilder().build(recipe_dir=recipe)
    assert len(Dataset("test", recipe_dir=recipe, condition="distant")) == 2
    for split in SPLITS:
        dataset = Dataset(split, recipe_dir=recipe)
        for index, row in enumerate(dataset.entries):
            sample = dataset[index]
            assert set(sample) == {"speech", "text"}
            assert sample["speech"].dtype == np.float32
            np.testing.assert_array_equal(
                sample["speech"], sf.read(row["path"], dtype="float32")[0]
            )


@pytest.mark.parametrize(
    "selectors", [{"split": "dev"}, {"split": "test", "condition": "unknown"}]
)
def test_reject_unknown_selectors(tmp_path, selectors):
    """Fail explicitly for unsupported splits or recording conditions."""
    with pytest.raises(ValueError, match="Unknown VOiCES"):
        Dataset(recipe_dir=tmp_path, **selectors)
