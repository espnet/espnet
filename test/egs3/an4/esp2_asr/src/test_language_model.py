"""Check LM corpus preparation independently of the shared LM trainer."""

import gzip

import pytest
from omegaconf import OmegaConf

from egs3.an4.esp2_asr.src.language_model import prepare_lm_text


@pytest.mark.parametrize("external", [False, True])
def test_prepare_lm_text_keeps_source_order_and_ids(tmp_path, external):
    """Keep pre-filter source transcripts and append external text only on request."""
    train = tmp_path / "train.txt"
    train.write_text("b HELLO WORLD\na AGAIN\nempty\n")
    archive = tmp_path / "external.gz"
    with gzip.open(archive, "wt") as stream:
        stream.write("EXTERNAL WORDS\n\nTHIRD LINE\n")
    config = OmegaConf.create(
        {
            "exp_dir": str(tmp_path / "lm"),
            "train_text": str(train),
            "external_text": (
                {"archive": str(archive), "url": "unused"} if external else None
            ),
        }
    )
    expected = ["b HELLO WORLD", "a AGAIN"]
    if external:
        expected += [
            "librispeech_lng_00000001 EXTERNAL WORDS",
            "librispeech_lng_00000003 THIRD LINE",
        ]
    path = prepare_lm_text(config)
    assert path.read_text().splitlines() == expected
    assert prepare_lm_text(config).read_text().splitlines() == expected


def test_empty_lm_text_does_not_replace_previous_output(tmp_path):
    """Fail clearly without overwriting the last usable text file."""
    train = tmp_path / "train.txt"
    train.write_text("empty\n")
    destination = tmp_path / "lm_train.txt"
    destination.write_text("previous DATA\n")
    config = OmegaConf.create({"exp_dir": str(tmp_path), "train_text": str(train)})
    with pytest.raises(RuntimeError, match="empty"):
        prepare_lm_text(config)
    assert destination.read_text() == "previous DATA\n"
