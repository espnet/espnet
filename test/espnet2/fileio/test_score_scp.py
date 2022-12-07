from pathlib import Path

import music21 as m21

from espnet2.fileio import NOTE, XMLReader  # SingingScoreReader, SingingScoreWriter


def test_XMLReader(tmp_path: Path):
    m = m21.stream.Stream()
    m.insert(m21.tempo.MetronomeMark(number=120))

    n = m21.note.Rest()
    n.offset = 0
    n.duration = m21.duration.Duration(1)
    m.inseert(n)

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
    assert val[1] == notes_list
