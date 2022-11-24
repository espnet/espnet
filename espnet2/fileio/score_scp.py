import collections.abc
from pathlib import Path
from typing import Dict, List, Union

import numpy as np
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2column_text

try:
    import music21 as m21  # for CI import
except ImportError or ModuleNotFoundError:
    m21 = None


class NOTE(object):
    def __init__(self, lyric, pitch, st, et, dur):
        self.lyric = lyric
        self.pitch = pitch
        self.st = st
        self.et = et
        self.dur = dur


class XMLReader(collections.abc.Mapping):
    """Reader class for 'xml.scp'.

    Examples:
        key1 /some/path/a.xml
        key2 /some/path/b.xml
        key3 /some/path/c.xml
        key4 /some/path/d.xml
        ...

        >>> reader = XMLScpReader('xml.scp')
        >>> lyrics_array, notes_array, segs_array = reader['key1']
    """

    def __init__(
        self,
        fname,
        dtype=np.int16,
    ):
        assert check_argument_types()
        assert m21 is not None, (
            "Cannot load music21 package. ",
            "Please install Muskit modules via ",
            "(cd tools && make muskit.done)",
        )
        self.fname = fname
        self.dtype = dtype
        self.data = read_2column_text(fname)  # get key-value dict

    def __getitem__(self, key):
        score = m21.converter.parse(self.data[key])
        m = score.metronomeMarkBoundaries()
        tempo = m[0][2].number
        part = score.parts[0].flat
        notes_list = []
        prepitch = -1
        st = 0
        for note in part.notesAndRests:
            if not note.isRest:  # Note or Chord
                lr = note.lyric
                if note.isChord:
                    for n in note:
                        if n.pitch.midi != prepitch:  # Ignore repeat note
                            note = n
                            break
                if lr is None or lr == "":  # multi note in one syllable
                    if note.pitch.midi == prepitch:  # same pitch
                        notes_list[-1].dur += note.seconds
                        notes_list[-1].et += note.seconds
                    else:  # different pitch
                        notes_list.append(
                            NOTE(
                                "—",
                                note.pitch.midi,
                                st,
                                st + note.seconds,
                                note.seconds,
                            )
                        )
                elif lr == "br":  # <br> is tagged as a note
                    if prepitch == 0:
                        notes_list[-1].dur += note.seconds
                        notes_list[-1].et += note.seconds
                    else:
                        notes_list.append(
                            NOTE("P", 0, st, st + note.seconds, note.seconds)
                        )
                    prepitch = 0
                else:  # normal note for one syllable
                    notes_list.append(
                        NOTE(
                            note.lyric,
                            note.pitch.midi,
                            st,
                            st + note.seconds,
                            note.seconds,
                        )
                    )
                prepitch = note.pitch.midi
                for arti in note.articulations:  # <br> is tagged as a notation
                    if arti.name in ["breath mark"]:  # up-bow?
                        notes_list.append(NOTE("B", 0, st, st, 0))
            else:  # rest note
                if prepitch == 0:
                    notes_list[-1].dur += note.seconds
                    notes_list[-1].et += note.seconds
                else:
                    notes_list.append(NOTE("P", 0, st, st + note.seconds, note.seconds))
                prepitch = 0
            st += note.seconds
        seq_len = len(notes_list)
        if notes_list[-1].pitch == 0:
            seq_len -= 1
        lyrics = []
        notes = np.zeros(seq_len, dtype=self.dtype)
        segs = np.zeros((seq_len, 2))
        for i in range(seq_len):
            segs[i, 0] = np.float32(notes_list[i].st)
            segs[i, 1] = np.float32(notes_list[i].et)
            notes[i] = notes_list[i].pitch
            lyrics.append(notes_list[i].lyric)
        return lyrics, notes, segs, tempo

    def get_path(self, key):
        return self.data[key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        return self.data.keys()


class XMLWriter:
    """Writer class for 'midi.scp'

    Examples:
        key1 /some/path/a.musicxml
        key2 /some/path/b.musicxml
        key3 /some/path/c.musicxml
        key4 /some/path/d.musicxml
        ...

        >>> writer = XMLScpWriter('./data/', './data/xml.scp')
        >>> writer['aa'] = xml_obj
        >>> writer['bb'] = xml_obj

    """

    def __init__(
        self,
        outdir: Union[Path, str],
        scpfile: Union[Path, str],
    ):
        assert check_argument_types()
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")
        self.data = {}

    def __setitem__(self, key: str, value: tuple):
        assert (
            len(value) == 4
        ), "The xml values should include lyrics, note, segmentations and tempo"
        lyrics_seq, notes_seq, segs_seq, tempo = value
        xml_path = self.dir / f"{key}.musicxml"
        xml_path.parent.mkdir(parents=True, exist_ok=True)

        m = m21.stream.Stream()
        m.insert(m21.tempo.MetronomeMark(number=tempo))
        bps = 1.0 * tempo / 60
        offset = 0
        for i in range(len(lyrics_seq)):
            duration = int(8 * (segs_seq[i][1] - segs_seq[i][0]) * bps + 0.5)
            duration = 1.0 * duration / 8
            if duration == 0:
                duration = 1 / 16
            if notes_seq[i] != -1:  # isNote
                n = m21.note.Note(notes_seq[i])
                if lyrics_seq[i] != "—":
                    n.lyric = lyrics_seq[i]
            else:  # isRest
                n = m21.note.Rest()
            n.offset = offset
            n.duration = m21.duration.Duration(duration)
            m.insert(n)
            offset += duration
        m.write("xml", fp=xml_path)
        self.fscp.write(f"{key} {xml_path}\n")
        self.data[key] = str(xml_path)

    def get_path(self, key):
        return self.data[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()


def read_score(path: Union[Path, str]) -> Dict[str, List[Union[float, int]]]:
    """Read a text file indicating sequences of number

    Examples:
        key1 start_time_1 end_time_1 phone_1 start_time_2 end_time_2 phone_2 ....\n
        key2 start_time_1 end_time_1 phone_1 \n

        >>> d = load_num_sequence_text('label')
        >>> np.testing.assert_array_equal(d["key1"], [0.1, 0.2, "啊"]))
    """
    assert check_argument_types()
    label = open(path, "r", encoding="utf-8")

    retval = {}
    for label_line in label.readlines():
        line = label_line.strip().split()
        key = line[0]
        tempo = int(float(line[1]))
        syb_info = line[2:]
        temp_info = []
        for i in range(len(syb_info) // 5):
            temp_info.append(
                [
                    float(syb_info[i * 5]),
                    float(syb_info[i * 5 + 1]),
                    syb_info[i * 5 + 2],
                    int(syb_info[i * 5 + 3]),
                    syb_info[i * 5 + 4],
                ]
            )
        retval[key] = tempo, temp_info
    return retval
