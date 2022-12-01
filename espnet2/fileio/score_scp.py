import collections.abc
import json
from pathlib import Path
from typing import Union

import numpy as np
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2column_text

try:
    import music21 as m21  # for CI import
except ImportError or ModuleNotFoundError:
    m21 = None


class NOTE(object):
    def __init__(self, lyric, midi, st, et):
        self.lyric = lyric
        self.midi = midi
        self.st = st
        self.et = et


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
        tempo = int(m[0][2].number)
        part = score.parts[0].flat
        notes_list = []
        prepitch = -1
        st = 0
        for note in part.notesAndRests:
            dur = note.seconds
            if not note.isRest:  # Note or Chord
                lr = note.lyric
                if note.isChord:
                    for n in note:
                        if n.pitch.midi != prepitch:  # Ignore repeat note
                            note = n
                            break
                if lr is None or lr == "":  # multi note in one syllable
                    if note.pitch.midi == prepitch:  # same pitch
                        notes_list[-1].et += dur
                    else:  # different pitch
                        notes_list.append(NOTE("—", note.pitch.midi, st, st + dur))
                elif lr == "br":  # <br> is tagged as a note
                    if prepitch == 0:
                        notes_list[-1].et += dur
                    else:
                        notes_list.append(NOTE("P", 0, st, st + dur))
                    prepitch = 0
                else:  # normal note for one syllable
                    notes_list.append(NOTE(note.lyric, note.pitch.midi, st, st + dur))
                prepitch = note.pitch.midi
                for arti in note.articulations:  # <br> is tagged as a notation
                    if arti.name in ["breath mark"]:  # up-bow?
                        notes_list.append(NOTE("B", 0, st, st))  # , 0))
            else:  # rest note
                if prepitch == 0:
                    notes_list[-1].et += dur
                else:
                    notes_list.append(NOTE("P", 0, st, st + dur))
                prepitch = 0
            st += dur
        return tempo, notes_list

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


class SingingScoreReader(collections.abc.Mapping):
    """Reader class for 'score.scp'.

    Examples:
        key1 /some/path/score.json
        key2 /some/path/score.json
        key3 /some/path/score.json
        key4 /some/path/score.json
        ...

        >>> reader = SoundScpReader('score.scp')
        >>> score = reader['key1']

    """

    def __init__(
        self,
        fname,
        dtype=np.int16,
    ):
        assert check_argument_types()
        self.fname = fname
        self.dtype = dtype
        self.data = read_2column_text(fname)

    def __getitem__(self, key):
        with open(self.data[key], "r") as f:
            score = json.load(f)
        return score

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


class SingingScoreWriter:
    """Writer class for 'score.scp'

    Examples:
        key1 /some/path/score.json
        key2 /some/path/score.json
        key3 /some/path/score.json
        key4 /some/path/score.json
        ...

        >>> writer = SingingScoreWriter('./data/', './data/score.scp')
        >>> writer['aa'] = score_obj
        >>> writer['bb'] = score_obj

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

    def __setitem__(self, key: str, value: dict):
        """Score should be a dict

        Example:
        {
            "tempo": bpm,
            "item_list": a subset of ["st", "et", "lyric", "midi", "phn"],
            "note": [
                [start_time1, end_time1, lyric1, midi1, phn1],
                [start_time2, end_time2, lyric2, midi2, phn2],
                ...
            ]
        }

        The itmes in each note correspond to the "item_list".

        """

        score_path = self.dir / f"{key}.json"
        score_path.parent.mkdir(parents=True, exist_ok=True)
        with open(score_path, "w") as f:
            json.dump(value, f, ensure_ascii=False, indent=2)
        self.fscp.write(f"{key} {score_path}\n")
        self.data[key] = str(score_path)

    def get_path(self, key):
        return self.data[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()
