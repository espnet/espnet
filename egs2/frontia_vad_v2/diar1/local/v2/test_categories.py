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
    if make is h_dump:  # the writer's own categories_rttm (an A dump's is refreshed
        # by align.py, which a_dump does not run)
        assert cat.main(["check", "--map", v2x, str(d)]) == 0
    # a dump written before categories_rttm existed: backfill
    (d / "categories_rttm").unlink()
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


def test_writer_categories_layers_and_packs(tmp_path):
    d = tmp_path / "w"
    w = VadDumpWriter(d, "toy", "meeting", "H", "CC0", pack_seconds=30)
    w.add(
        "talk",
        noise(40),
        FS,
        speech=[(1.0, 3.0, "laughs"), (2.5, 4.0), (10.0, 12.0)],
        categories=[(3.2, 3.6, "vocal/laughter", "ami:laugh")],
        details=[(10.5, 11.0, "speech/filler", "uh")],
    )
    w.add("song", noise(8), FS, clip_category=("music/music", "openmic"))
    w.add("cough", noise(8, 1), FS, negative_type="cough")
    w.close()
    segs, lengths = cat.read_categories(d / "categories_rttm")
    talk = sorted(
        (s.layer, s.start, s.end, s.category, s.tags)
        for s in segs
        if s.rec == "toy-talk"
    )
    assert talk == [
        (1, 1 * FS, 4 * FS, "speech/utterance", "laughs"),
        (1, int(3.2 * FS), int(3.6 * FS), "vocal/laughter", "ami:laugh"),
        (1, 10 * FS, 12 * FS, "speech/utterance", "<NA>"),
        (2, int(10.5 * FS), 11 * FS, "speech/filler", "uh"),
    ]
    # the two short clips share a pack, each with its class at its offset
    pack = sorted((s.start, s.category) for s in segs if s.rec == "toy-song+2")
    assert pack == [(0, "music/music"), (8 * FS, "vocal/cough")]
    v2x = cat.load_map(MAPS / "v2x.yaml")
    assert cat.apply_map(segs, lengths, v2x)[0] == (d / "espnet_rttm").read_text()
    # the detail layer counts only for a map that asks for it
    p = tmp_path / "details.yaml"
    p.write_text(
        "classes: [speech, filler]\ndetails: true\nmap:\n"
        '  - "speech/filler": filler\n  - "speech/*": speech\n  - "*": nonspeech\n'
    )
    text, _ = cat.apply_map(segs, lengths, cat.load_map(p))
    assert "toy-talk 1 168000 176000 <NA> <NA> filler" in text


def test_writer_refuses_speech_in_non_speech_slots(tmp_path):
    w = VadDumpWriter(tmp_path / "x", "toy", "meeting", "H", "CC0")
    with pytest.raises(ValueError, match="speech goes in"):
        w.add("a", noise(2), FS, categories=[(0, 1, "speech/words", "")])
    with pytest.raises(ValueError, match="unknown category"):
        w.add("b", noise(2), FS, details=[(0, 1, "vocal/yodel", "")])
    with pytest.raises(ValueError, match="clip_category"):
        w.add("c", noise(2), FS, clip_category=("speech/singing", ""))


def test_alignment_refresh_keeps_classes_and_drops_recordings(tmp_path):
    d = tmp_path / "a"
    w = VadDumpWriter(d, "toya", "read", "A", "CC0")
    for k in range(3):
        w.add(
            f"u{k}",
            noise(12, k),
            FS,
            text="a b",
            lang_sym="<eng>",
            categories=[(5.0, 6.0, "vocal/breath", "")],
        )
    w.close()
    # what align.py writes for the recordings it keeps
    (d / "espnet_rttm").write_text(
        "END toya-u0 <NA> <NA> 192000 <NA> <NA> <NA> <NA>\n"
        "END toya-u1 <NA> <NA> 192000 <NA> <NA> <NA> <NA>\n"
        "SPEAKER toya-u0 1 1600 64000 <NA> <NA> toya-u0 <NA>\n"
        "SPEAKER toya-u1 1 3200 48000 <NA> <NA> toya-u1 <NA>\n"
    )
    cat.refresh_speech(d, "speech/words", keep={"toya-u0", "toya-u1"})
    segs, lengths = cat.read_categories(d / "categories_rttm")
    assert set(lengths) == {"toya-u0", "toya-u1"}
    assert sorted((s.rec, s.category) for s in segs) == [
        ("toya-u0", "speech/words"),
        ("toya-u0", "vocal/breath"),
        ("toya-u1", "speech/words"),
        ("toya-u1", "vocal/breath"),
    ]
    assert cat.check(d, cat.load_map(MAPS / "v2x.yaml"))
