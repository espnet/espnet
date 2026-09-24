"""Verify source text coverage before ASR duration filtering."""

import numpy as np
import soundfile as sf

from egs3.voices.esp2_asr.dataset import DatasetBuilder
from egs3.voices.esp2_asr.dataset.builder import read_manifest
from egs3.voices.esp2_asr.src.tokenizer import gather_training_text


def test_tokenizer_and_lm_keep_pre_filter_text(corpus):
    """Duration filtering must not remove source tokenizer or LM training text."""
    recipe, source = corpus
    path = sorted((source / "source-16k/train").rglob("*.wav"))[-1]
    sf.write(path, np.zeros(480000), 16000)
    DatasetBuilder().build(recipe_dir=recipe)
    assert len(read_manifest(recipe / "data/manifest/train.tsv")) == 5
    assert len(gather_training_text(recipe / "data/manifest/tokenizer_train.txt")) == 6
    assert len((recipe / "data/lm/train.txt").read_text().splitlines()) == 6
