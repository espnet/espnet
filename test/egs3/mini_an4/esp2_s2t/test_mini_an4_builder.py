"""Tests for the mini AN4 SOT builder in egs3/mini_an4/esp2_s2t/dataset/.

Everything here runs off the archive committed in the recipe, so these need no
corpus and no network. What they pin is the part that is not shared with the
AMI recipe: AN4 is single-speaker, and this builder is what turns it into
multi-talker utterance groups.
"""

import sys
from pathlib import Path

import pytest

pytest.importorskip("lhotse")
pytest.importorskip("soundfile")

_REPO = Path(__file__).resolve().parents[4]
_RECIPE = _REPO / "egs3" / "mini_an4" / "esp2_s2t"
sys.path.insert(0, str(_REPO))

from egs3.mini_an4.esp2_s2t.dataset import builder as bl  # noqa: E402


def test_parse_transcript_drops_the_sentence_markers_and_lowercases(tmp_path):
    """AN4 writes `<s> WORDS </s> (utt-id)`; none of that belongs in a target."""
    path = tmp_path / "an4.transcription"
    path.write_text(
        "<s> YES </s> (an251-fash-b)\nELEVEN TWENTY SEVEN (cen8-fcaw-b)\n",
        encoding="utf-8",
    )
    assert bl.parse_transcript(path) == {
        "an251-fash-b": "yes",
        "cen8-fcaw-b": "eleven twenty seven",
    }


@pytest.mark.execution_timeout(60)
def test_build_makes_multi_speaker_groups_out_of_single_speaker_audio(
    tmp_path, monkeypatch
):
    """The whole point of this builder.

    AN4 has one speaker per utterance. A serialized-output-training corpus
    needs several per group, so the builder concatenates utterances from
    different speakers and writes one cut over the result. Without that there
    is nothing for the separator to separate and the recipe would exercise
    none of the SOT path.
    """
    monkeypatch.setenv("MINI_AN4_SOT_DATA_ROOT", str(tmp_path))
    # The archive lives beside the recipe, not beside the corpus root.
    monkeypatch.setitem(bl._CONFIG, "archive_path", str(_RECIPE / "downloads.tar.gz"))

    bl.MiniAn4SotBuilder().build()

    from lhotse import CutSet

    cuts = list(CutSet.from_file(str(tmp_path / "data" / "test" / "cuts.jsonl.gz")))
    assert len(cuts) == 1
    cut = cuts[0]
    speakers = {supervision.speaker for supervision in cut.supervisions}
    assert len(speakers) > 1, f"one speaker only: {speakers}"

    text = cut.custom[bl._SOT_TEXT_FIELD]
    assert bl._CONFIG["separator"] in text
    assert text.startswith(bl._CONFIG["prompt"])
    # The supervisions are laid end to end, so the second block opens after the
    # first one closes.
    starts = sorted(supervision.start for supervision in cut.supervisions)
    assert starts[1] > starts[0]


@pytest.mark.execution_timeout(60)
def test_the_token_list_holds_the_separator_it_serialized_with(tmp_path, monkeypatch):
    """The exported vocabulary and the corpus have to agree on the symbol.

    A token list without the separator at the id the tokenizer resolves it to
    would train the model against a row it can never emit.
    """
    monkeypatch.setenv("MINI_AN4_SOT_DATA_ROOT", str(tmp_path))
    monkeypatch.setitem(bl._CONFIG, "archive_path", str(_RECIPE / "downloads.tar.gz"))

    bl.MiniAn4SotBuilder().build()

    tokens = (tmp_path / "data" / "tokens.txt").read_text().splitlines()
    separator = bl._CONFIG["separator"]
    assert tokens.count(separator) == 1
    assert tokens.index(separator) == 25629
