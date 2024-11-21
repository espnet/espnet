import json
from pathlib import Path

import miditoolkit
import miditoolkit.midi.containers as ct
import music21 as m21

from espnet2.fileio.score_scp import (
    NOTE,
    MIDReader,
    SingingScoreReader,
    SingingScoreWriter,
    XMLReader,
    XMLWriter,
)


def test_XMLReader(tmp_path: Path):
    m = m21.stream.Stream()
    m.insert(m21.tempo.MetronomeMark(number=60))

    n = m21.note.Rest()
    n.offset = 0
    n.duration = m21.duration.Duration(1)
    m.insert(n)

    n = m21.note.Note(20)
    n.lyric = "a"
    n.offset = 1
    n.duration = m21.duration.Duration(2)
    m.insert(n)

    n = m21.note.Note(21)
    n.lyric = ""
    n.offset = 3
    n.duration = m21.duration.Duration(1)
    m.insert(n)

    n = m21.note.Note(22)
    n.lyric = "ー"
    n.offset = 4
    n.duration = m21.duration.Duration(1)
    m.insert(n)

    n = m21.note.Note(23)
    n.lyric = None
    n.offset = 5
    n.duration = m21.duration.Duration(2)
    m.insert(n)

    n = m21.note.Note(24)
    n.lyric = "b"
    n.offset = 7
    n.duration = m21.duration.Duration(1)
    n.articulations.append(m21.articulations.BreathMark())
    m.insert(n)

    n = m21.chord.Chord([24, 25])
    n.lyric = "c"
    n.offset = 8
    n.duration = m21.duration.Duration(1)
    m.insert(n)

    n = m21.note.Note(25)
    n.lyric = "br"
    n.offset = 9
    n.duration = m21.duration.Duration(1)
    m.insert(n)

    n = m21.note.Rest()
    n.offset = 10
    n.duration = m21.duration.Duration(1)
    m.insert(n)

    n = m21.note.Note(25)
    n.lyric = "br"
    n.offset = 11
    n.duration = m21.duration.Duration(1)
    m.insert(n)

    n = m21.note.Note(26)
    n.lyric = "d"
    n.offset = 12
    n.duration = m21.duration.Duration(1)
    m.insert(n)

    xml_path = tmp_path / "abc.xml"
    m.write("xml", fp=xml_path)

    p = tmp_path / "dummy3.scp"
    with p.open("w") as f:
        f.write(f"abc {xml_path}\n")

    print(p)
    reader = XMLReader(p)
    val = reader["abc"]

    assert len(val) == 2
    assert val[0] == 60
    notes_list = [
        NOTE("P", 0, 0, 1),
        NOTE("a", 20, 1, 3),
        NOTE("—", 21, 3, 4),
        NOTE("—", 22, 4, 5),
        NOTE("—", 23, 5, 7),
        NOTE("b", 24, 7, 8),
        NOTE("B", 0, 8, 8),
        NOTE("c", 25, 8, 9),
        NOTE("P", 0, 9, 12),
        NOTE("d", 26, 12, 13),
    ]
    for i in range(len(val[1])):
        assert val[1][i].st == notes_list[i].st
        assert val[1][i].et == notes_list[i].et
        assert val[1][i].lyric == notes_list[i].lyric
        assert val[1][i].midi == notes_list[i].midi


def test_XMLWriter(tmp_path: Path):
    lyrics_seq = ["P", "a", "—", "b"]
    notes_seq = [0, 20, 21, 22]
    segs_seq = [(0, 1), (1, 2), (2, 3), (3, 3.1)]
    tempo = 60
    val = lyrics_seq, notes_seq, segs_seq, tempo
    p = tmp_path / "dummy4.scp"
    with XMLWriter(tmp_path, p) as writer:
        writer["abc"] = val

    reader = XMLReader(p)
    val = reader["abc"]
    assert len(val) == 2
    assert val[0] == 60
    notes_list = [
        NOTE("P", 0, 0, 1),
        NOTE("a", 20, 1, 2),
        NOTE("—", 21, 2, 3),
        NOTE("b", 22, 3, 3.125),
    ]
    for i in range(len(val[1])):
        assert val[1][i].st == notes_list[i].st
        assert val[1][i].et == notes_list[i].et
        assert val[1][i].lyric == notes_list[i].lyric
        assert val[1][i].midi == notes_list[i].midi


def test_MIDReader(tmp_path: Path):
    mido_obj = miditoolkit.midi.parser.MidiFile()
    mido_obj.tempo_changes.append(ct.TempoChange(60, 0))
    track = ct.Instrument(program=0, is_drum=False, name="example track")
    mido_obj.instruments = [track]
    beat_resol = mido_obj.ticks_per_beat

    st = beat_resol * 1
    et = beat_resol * 2
    mido_obj.instruments[0].notes.append(
        ct.Note(start=st, end=et, pitch=20, velocity=100)
    )

    st = beat_resol * 3
    et = beat_resol * 4
    mido_obj.instruments[0].notes.append(
        ct.Note(start=st, end=et, pitch=21, velocity=100)
    )

    st = beat_resol * 4
    et = beat_resol * 5
    mido_obj.instruments[0].notes.append(
        ct.Note(start=st, end=st, pitch=22, velocity=100)
    )

    mid_path = tmp_path / "abc.mid"
    mido_obj.dump(mid_path)

    p = tmp_path / "dummy0.scp"
    with p.open("w") as f:
        f.write(f"abc {mid_path}\n")

    reader = MIDReader(p, add_rest=True)
    val = reader["abc"]

    assert len(val) == 2
    assert val[0] == 60
    notes_list = [
        NOTE("P", 0, 0, 1),
        NOTE("*", 20, 1, 2),
        NOTE("P", 0, 2, 3),
        NOTE("*", 21, 3, 4),
        NOTE("*", 22, 4, 5),
    ]
    for i in range(len(val[1])):
        assert val[1][i].st == notes_list[i].st
        assert val[1][i].et == notes_list[i].et
        assert val[1][i].lyric == notes_list[i].lyric
        assert val[1][i].midi == notes_list[i].midi

    reader = MIDReader(p, add_rest=False)
    val = reader["abc"]

    assert len(val) == 2
    assert val[0] == 60
    notes_list = [NOTE("*", 20, 1, 2), NOTE("*", 21, 3, 4), NOTE("*", 22, 4, 5)]
    for i in range(len(val[1])):
        assert val[1][i].st == notes_list[i].st
        assert val[1][i].et == notes_list[i].et
        assert val[1][i].lyric == notes_list[i].lyric
        assert val[1][i].midi == notes_list[i].midi


def test_SingingScoreReader(tmp_path: Path):
    dic = {}
    tempo = 100
    item_list = ["st", "et", "lyric", "midi", "phn"]
    notes_list = [[0.0, 1.0, "a", 20, "a"], [1.0, 2.2, "b_c", 22, "b_c"]]
    dic.update(tempo=tempo, item_list=item_list, note=notes_list)

    score_path = tmp_path / "abc.json"
    with open(score_path, "w") as f:
        json.dump(dic, f, ensure_ascii=False, indent=2)

    p = tmp_path / "dummy1.scp"
    with p.open("w") as f:
        f.write(f"abc {score_path}\n")

    reader = SingingScoreReader(p)
    val = reader["abc"]

    assert val == dic


def test_SingingScoreWriter(tmp_path: Path):
    dic = {}
    tempo = 100
    item_list = ["st", "et", "lyric", "midi", "phn"]
    notes_list = [[0.0, 1.0, "a", 20, "a"], [1.0, 2.2, "b_c", 22, "b_c"]]
    dic.update(tempo=tempo, item_list=item_list, note=notes_list)

    p = tmp_path / "dummy2.scp"
    with SingingScoreWriter(tmp_path, p) as writer:
        writer["abc"] = dic

    reader = SingingScoreReader(p)
    val = reader["abc"]

    assert val == dic
