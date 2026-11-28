"""Tests of the category labels (local/v2/categories.py) on vad_dump.py dumps.

python3 -m pytest local/v2/test_categories.py
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import categories as cat  # noqa: E402
from vad_dump import VadDumpWriter  # noqa: E402

MAPS = Path(__file__).resolve().parents[2] / "conf" / "label_map"
FS = 16000


def noise(seconds, seed=0):
    return np.random.default_rng(seed).standard_normal(int(seconds * FS)) * 0.01


def h_dump(tmp_path):
    """Speech with human segments, a two-channel recording, packed non-speech clips."""
    d = tmp_path / "h"
    w = VadDumpWriter(d, "toy", "meeting", "H", "CC0", pack_seconds=30)
    w.add("talk", noise(40), FS, speech=[(1.0, 3.5), (3.5, 4.0), (10.2, 12.71)])
    w.add("stereo", np.stack([noise(35), noise(35, 1)], 1), FS, speech=[(2, 5)])
    for k, nt in enumerate(["cough", "Dog", "music_fma", "cough", "laugh"]):
        w.add(f"clip{k}", noise(8, k), FS, negative_type=nt)
    w.close()
    return d


def a_dump(tmp_path):
    d = tmp_path / "a"
    w = VadDumpWriter(d, "toya", "read", "A", "CC0", pack_seconds=30)
    for k in range(4):
        w.add(f"u{k}", noise(12, k), FS, text=f"word {k}", lang_sym="<eng>")
    w.close()
    # what align.py leaves: speech intervals in espnet_rttm
    rttm = []
    for line in (d / "espnet_rttm").read_text().splitlines():
        f = line.split()
        if f[0] == "END":
            n = int(f[4])
            rttm += [
                f"SPEAKER {f[1]} 1 1600 32000 <NA> <NA> {f[1]} <NA>\n",
                f"SPEAKER {f[1]} 1 {n - 32000} {n - 16000} <NA> <NA> {f[1]} <NA>\n",
            ]
        rttm.append(line + "\n")
    (d / "espnet_rttm").write_text("".join(sorted(rttm)))
    return d


@pytest.mark.parametrize("make", [h_dump, a_dump])
def test_default_map_gives_espnet_rttm_byte_for_byte(tmp_path, make):
    d = make(tmp_path)
    v2x = str(MAPS / "v2x.yaml")
    assert cat.main(["selfcheck", "--map", v2x, str(d)]) == 0
    assert not (d / "categories_rttm").exists()  # selfcheck writes nothing
    assert cat.main(["backfill", str(d)]) == 0
    assert cat.main(["check", "--map", str(MAPS / "v2x.yaml"), str(d)]) == 0
    segs, _ = cat.read_categories(d / "categories_rttm")
    speech = {s.category for s in segs if s.category.startswith("speech/")}
    assert speech == ({"speech/words"} if make is a_dump else {"speech/utterance"})


def test_pack_members_keep_each_clip_class(tmp_path):
    d = h_dump(tmp_path)
    members = cat.read_pack_members(d / "pack_members")
    # a full pack of 32 s, and the last clip flushed alone at close
    assert sorted(members) == ["toy-clip0+4", "toy-clip4+1"]
    pack = "toy-clip0+4"
    assert [m[3] for m in members[pack]] == ["cough", "Dog", "music_fma", "cough"]
    segs, _ = cat.backfill_segments(d)
    in_pack = sorted((s.start, s.category, s.tags) for s in segs if s.rec == pack)
    assert in_pack == [
        (0, "vocal/cough", "cough"),
        (8 * FS, "sound/animal", "Dog"),
        (16 * FS, "music/music", "music_fma"),
        (24 * FS, "vocal/cough", "cough"),
    ]
    (last,) = [s for s in segs if s.tags == "laugh"]
    assert last.category == "vocal/laughter" and last.rec == "toy-clip4+1"


def test_a_change_to_the_categories_is_caught(tmp_path):
    d = h_dump(tmp_path)
    cat.main(["backfill", str(d)])
    lines = (d / "categories_rttm").read_text().splitlines()
    i = next(k for k, line in enumerate(lines) if "speech/utterance" in line)
    lines[i] = lines[i].replace("speech/utterance", "vocal/laughter")
    (d / "categories_rttm").write_text("\n".join(lines) + "\n")
    assert cat.main(["check", "--map", str(MAPS / "v2x.yaml"), str(d)]) == 1


def test_four_classes_and_the_evaluation_map(tmp_path):
    d = h_dump(tmp_path)
    cat.main(["backfill", str(d)])
    segs, lengths = cat.read_categories(d / "categories_rttm")
    four = cat.load_map(MAPS / "classes4.yaml")
    text, ignore = cat.apply_map(segs, lengths, four)
    spk = {line.split()[7] for line in text.splitlines() if line.startswith("SPEAKER")}
    assert spk == {"speech", "vocal", "music", "sound"} and not ignore
    ev = cat.load_map(MAPS / "eval_vocal_ignore.yaml")
    text, ignore = cat.apply_map(segs, lengths, ev)
    assert {s.category for s in ignore} == {"vocal/cough", "vocal/laughter"}
    assert "music" not in text


def test_maps_must_decide_every_category(tmp_path):
    p = tmp_path / "m.yaml"
    p.write_text('map:\n  - "speech/*": speech\n')
    with pytest.raises(ValueError, match="no rule"):
        cat.load_map(p)
    p.write_text('map:\n  - "*": loud\n')
    with pytest.raises(ValueError, match="not in"):
        cat.load_map(p)
    p.write_text('classes: [speech, ignore]\nmap:\n  - "*": speech\n')
    with pytest.raises(ValueError, match="bad classes"):
        cat.load_map(p)


def test_unknown_category_in_a_file_is_refused(tmp_path):
    p = tmp_path / "categories_rttm"
    p.write_text("SPEAKER r 1 0 10 <NA> <NA> vocal/yodel <NA>\n")
    with pytest.raises(ValueError, match="unknown category"):
        cat.read_categories(p)


@pytest.mark.parametrize(
    "nt, category",
    [
        ("vocal_imitation:Animal", "vocal/vocal_imitation"),
        ("designed_vocalization:breath", "vocal/other_vocal"),
        ("caf", "sound/babble"),
        ("synthetic_silence", "sound/room_tone"),
        ("synthetic_pink", "sound/synthetic_noise"),
        ("keyboard_mechanical", "sound/household"),
        ("animal_PB", "sound/animal"),
        ("Crying_and_sobbing", "vocal/cry"),
        ("Dishes_and_pots_and_pans", "sound/household"),
        ("Acoustic_guitar", "music/music"),
        ("Rain", "sound/environment"),
        ("something_new", "sound/other_sound"),
    ],
)
def test_negative_type_categories(nt, category):
    assert cat.negative_type_category(nt) == category
