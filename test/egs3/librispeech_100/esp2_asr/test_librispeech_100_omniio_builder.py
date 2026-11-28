"""Tests for the omniio dataset/builder of the LibriSpeech 100h recipe."""

import io
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

pytest.importorskip("omniio")
pytest.importorskip("duckdb")

from egs3.librispeech_100.esp2_asr.dataset.omniio import (  # noqa: E402
    Dataset,
    DatasetBuilder,
)
from egs3.librispeech_100.esp2_asr.dataset.omniio.builder import (  # noqa: E402
    SPLITS,
)
from egs3.librispeech_100.esp2_asr.dataset.omniio.tokenizer import (  # noqa: E402
    gather_training_text,
)


def _flac_bytes(num_samples: int) -> bytes:
    """Return a silent 16 kHz FLAC file of ``num_samples`` samples."""
    buf = io.BytesIO()
    sf.write(buf, np.zeros(num_samples, dtype=np.float32), 16000, format="FLAC")
    return buf.getvalue()


def _write_tree(librispeech_root: Path) -> None:
    """Write two utterances per split, with a blank line in the transcript."""
    for split in SPLITS:
        chapter = librispeech_root / split / "19" / "198"
        chapter.mkdir(parents=True)
        (chapter / "19-198.trans.txt").write_text(
            "19-198-0000 HELLO WORLD\n\n19-198-0001 GOOD BYE\n"
        )
        (chapter / "19-198-0000.flac").write_bytes(_flac_bytes(1600))
        (chapter / "19-198-0001.flac").write_bytes(_flac_bytes(3200))


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    """Return (recipe_dir, source_dir, dump_root) with a toy LibriSpeech tree."""
    monkeypatch.delenv("OMNIIO_DUMP_ROOT", raising=False)
    monkeypatch.delenv("LIBRISPEECH", raising=False)
    source = tmp_path / "LibriSpeech"
    _write_tree(source)
    recipe_dir = tmp_path / "recipe"
    recipe_dir.mkdir()
    return recipe_dir, source, tmp_path / "dump"


def test_build_and_read_dump(corpus):
    """create_dataset's builder dumps every split; the dataset reads it back."""
    recipe_dir, source, dump_root = corpus
    builder = DatasetBuilder()
    kwargs = dict(
        recipe_dir=recipe_dir, source_dir=source, dump_root=dump_root, num_workers=0
    )
    assert builder.is_source_prepared(**kwargs)
    assert not builder.is_built(**kwargs)
    builder.build(**kwargs)
    assert builder.is_built(**kwargs)

    ds = Dataset(split="dev-clean", dump_root=dump_root)
    assert ds.keys() == ["19-198-0000", "19-198-0001"]
    assert ds[1]["text"] == "GOOD BYE" and ds[1]["speech"].shape == (3200,)
    assert ds.get_item_lengths().tolist() == [1600, 3200]
    texts = gather_training_text(dump_root=dump_root, split="train-clean-100")
    assert texts == ["HELLO WORLD", "GOOD BYE"]


def test_finished_dump_needs_no_source(corpus):
    """With the dump built, the raw corpus is no longer required."""
    recipe_dir, source, dump_root = corpus
    builder = DatasetBuilder()
    builder.build(
        recipe_dir=recipe_dir, source_dir=source, dump_root=dump_root, num_workers=0
    )
    assert builder.is_source_prepared(
        recipe_dir=recipe_dir, source_dir=recipe_dir / "missing", dump_root=dump_root
    )


def test_build_without_source_fails(corpus):
    """Building with no corpus and no dump reports the missing source."""
    recipe_dir, _, dump_root = corpus
    builder = DatasetBuilder()
    assert not builder.is_source_prepared(recipe_dir=recipe_dir, dump_root=dump_root)
    with pytest.raises(FileNotFoundError, match="LibriSpeech source not found"):
        builder.build(recipe_dir=recipe_dir, dump_root=dump_root, num_workers=0)
