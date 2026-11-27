"""Tests for the mini AN4 SOT builder in egs3/mini_an4/esp2_s2t/dataset/.

These tests use the archive committed in the recipe, so they need no corpus and
no network. They check the part that the AMI recipe does not share. AN4 has one
speaker per utterance, and this builder turns it into multi-talker utterance
groups.
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
    """parse_transcript drops the sentence markers and lowercases the words.

    AN4 writes `<s> WORDS </s> (utt-id)`. The markers and the id do not belong in a
    target.
    """
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
    """build() makes groups with several speakers from single-speaker AN4 audio.

    AN4 has one speaker per utterance. A serialized output training corpus needs
    several speakers per group, so the builder concatenates utterances from
    different speakers and writes one cut for the result. Without this step, the
    separator would have nothing to separate, and the recipe would run none of the
    SOT code path.
    """
    monkeypatch.setenv("MINI_AN4_SOT_DATA_ROOT", str(tmp_path))
    # The archive is in the recipe directory. The corpus root here is tmp_path.
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
    # The supervisions are placed one after another, so the second block starts
    # after the first one ends.
    starts = sorted(supervision.start for supervision in cut.supervisions)
    assert starts[1] > starts[0]


@pytest.mark.execution_timeout(60)
def test_the_token_list_holds_the_separator_it_serialized_with(tmp_path, monkeypatch):
    """The token list has the separator at the id that the tokenizer gives it.

    Otherwise decoding turns that id into another token, and the hypothesis
    has no separator.
    """
    monkeypatch.setenv("MINI_AN4_SOT_DATA_ROOT", str(tmp_path))
    monkeypatch.setitem(bl._CONFIG, "archive_path", str(_RECIPE / "downloads.tar.gz"))

    bl.MiniAn4SotBuilder().build()

    tokens = (tmp_path / "data" / "tokens.txt").read_text().splitlines()
    separator = bl._CONFIG["separator"]
    assert tokens.count(separator) == 1
    assert tokens.index(separator) == 25629


def test_one_variable_sets_the_separator_everywhere(monkeypatch):
    """MINI_AN4_SOT_SPEAKER_CHANGE_SYMBOL sets the symbol in every config."""
    from espnet3.utils.config_utils import (
        load_and_merge_config,
        load_config_with_defaults,
    )

    monkeypatch.setenv("MINI_AN4_SOT_SPEAKER_CHANGE_SYMBOL", "<sc>")

    def load(name):
        return load_and_merge_config(
            _RECIPE / "conf" / name,
            config_name=name,
            default_package="egs3.TEMPLATE.esp2_s2t",
            resolve=True,
        )

    dataset = load_config_with_defaults(str(_RECIPE / "dataset" / "config.yaml"))
    found = [
        dataset["builder"]["separator"],
        load("training.yaml").dataset.preprocessor.speaker_change_symbol,
        load("inference.yaml").model.speaker_change_symbol,
        *[entry.metric.speaker_change_symbol for entry in load("metrics.yaml").metrics],
    ]
    assert found == ["<sc>"] * len(found)
