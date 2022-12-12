import json
from pathlib import Path

import music21 as m21

from espnet2.fileio.score_scp import (
    NOTE,
    SingingScoreReader,
    SingingScoreWriter,
    XMLReader,
)


def test_XMLReader(tmp_path: Path):
    m = m21.stream.Stream()
    m.insert(m21.tempo.MetronomeMark(number=120))

    n = m21.note.Rest()
    n.offset = 0
    n.duration = m21.duration.Duration(1)
    m.insert(n)

    n = m21.note.Note(20)
    n.lyric = "a"
    n.offset = 1
    n.duration = m21.duration.Duration(2)
    m.insert(n)

    xml_path = tmp_path / "abc.xml"
    m.write("xml", fp=xml_path)

    p = tmp_path / "dummy.scp"
    with p.open("w") as f:
        f.write(f"abc {xml_path}\n")

    reader = XMLReader(p)
    val = reader["abc"]

    assert len(val) == 2
    assert val[0] == 120
    notes_list = [NOTE("P", 0, 0, 0.5), NOTE("a", 20, 0.5, 1.5)]
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
