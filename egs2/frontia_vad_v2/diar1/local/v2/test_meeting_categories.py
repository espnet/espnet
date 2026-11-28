"""Tests of the meeting category labels: transcript tags, NXT words XML, from_kaldi.

python3 -m pytest local/v2/test_meeting_categories.py
"""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

import categories as cat  # noqa: E402
from vad_data_utils import (  # noqa: E402
    nxt_words_and_sounds,
    tag_only_categories,
    transcript_info,
    write_vad_dir,
)


@pytest.mark.parametrize(
    "text, words, tags",
    [
        ("[laughs]", False, ["laughs"]),
        ("hello [noise] there", True, ["noise"]),
        ("<FILLlaugh/>", False, ["filllaugh"]),
        ("<BA/>", False, ["ba"]),
        ("[inaudible 0:12:31.3]", True, ["inaudible"]),
        ("[unintelligible] [laugh]", True, ["unintelligible", "laugh"]),
        ("", False, []),
    ],
)
def test_transcript_info(text, words, tags):
    assert transcript_info(text) == (words, tags)


def test_tag_only_categories():
    assert tag_only_categories(1.0, 2.0, ["laughs", "noise", "ba"]) == [
        (1.0, 2.0, "vocal/laughter", ["laughs"], 1),
        (1.0, 2.0, "vocal/speaker_noise", ["noise"], 1),
    ]


def test_nxt_words_and_sounds_places_untimed_sounds():
    xml = (
        '<w nite:id="a" starttime="1.0" endtime="1.5">hi</w>'
        '<w nite:id="p" starttime="1.5" endtime="1.5" punc="true">.</w>'
        '<vocalsound nite:id="v" starttime="" endtime="" description="breath"/>'
        '<w nite:id="b" starttime="2.0" endtime="2.4">there</w>'
        '<vocalsound nite:id="l" starttime="3.0" endtime="3.8" type="laugh"/>'
        '<nonvocalsound nite:id="n" starttime="4.0" endtime="4.2" type="door"/>'
    )
    words, sounds = nxt_words_and_sounds(xml)
    assert words == [(1.0, 1.5), (2.0, 2.4)]
    assert sounds == [
        (1.5, 2.0, "vocal/breath", ["breath", "untimed"], 1),
        (3.0, 3.8, "vocal/laughter", ["laugh"], 1),
        (4.0, 4.2, "sound/other_sound", ["nonvocal", "door"], 1),
    ]


def kaldi_dir(tmp_path, with_labels: bool):
    wav = tmp_path / "rec.wav"
    soundfile.write(wav, np.zeros(16000 * 10, dtype=np.float32), 16000)
    out = tmp_path / ("labelled" if with_labels else "plain")
    speech = {"m_rec": [(1.0, 2.0), (3.0, 4.5), (6.0, 7.0)]}
    utts = {
        "m_rec": [
            (1.0, 2.0, True, ["noise"]),
            (3.0, 3.5, False, ["laughs"]),
            (3.4, 4.5, True, []),
            (6.0, 7.0, False, ["ba"]),
        ]
    }
    cats = {
        "m_rec": [
            (3.0, 3.5, "vocal/laughter", ["laughs"], 1),
            (1.2, 1.6, "speech/words", [], 2),
        ]
    }
    if with_labels:
        write_vad_dir(out, {"m_rec": str(wav)}, speech, utts, cats)
    else:
        write_vad_dir(out, {"m_rec": str(wav)}, speech)
    return out


def from_kaldi(data_dir, dump):
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    subprocess.run(
        [
            sys.executable,
            str(HERE / "from_kaldi.py"),
            str(data_dir),
            str(dump),
            "--corpus",
            "m",
            "--domain",
            "meeting",
            "--licence",
            "CC0",
        ],
        check=True,
        env=env,
    )


def test_categories_through_from_kaldi_keep_espnet_rttm(tmp_path):
    plain, labelled = kaldi_dir(tmp_path, False), kaldi_dir(tmp_path, True)
    tags = (labelled / "segment_tags").read_text().splitlines()
    assert [t.split()[1:] for t in tags] == [
        ["speech/utterance", "noise"],
        ["speech/utterance", "laughs"],
        ["speech/blank", "ba"],
    ]
    from_kaldi(plain, tmp_path / "d_plain")
    from_kaldi(labelled, tmp_path / "d_labelled")
    a = (tmp_path / "d_plain" / "espnet_rttm").read_text()
    b = (tmp_path / "d_labelled" / "espnet_rttm").read_text()
    assert a == b
    v2x = cat.load_map(HERE.parents[1] / "conf" / "label_map" / "v2x.yaml")
    assert cat.check(tmp_path / "d_labelled", v2x)
    segs, _ = cat.read_categories(tmp_path / "d_labelled" / "categories_rttm")
    got = sorted((s.layer, s.start, s.category, s.tags) for s in segs)
    assert got == [
        (1, 16000, "speech/utterance", "noise"),
        (1, 48000, "speech/utterance", "laughs"),
        (1, 48000, "vocal/laughter", "laughs"),
        (1, 96000, "speech/blank", "ba"),
        (2, 19200, "speech/words", "<NA>"),
    ]
    # the evaluation map leaves out the laugh and the blank utterance
    ev = cat.load_map(HERE.parents[1] / "conf" / "label_map" / "eval_vocal_ignore.yaml")
    _, ignore = cat.apply_map(segs, {}, ev)
    assert sorted(s.category for s in ignore) == ["speech/blank", "vocal/laughter"]
