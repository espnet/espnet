import collections.abc
import json
from pathlib import Path
from typing import Union

import numpy as np
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2columns_text

try:
    import music21 as m21  # for CI import
except (ImportError, ModuleNotFoundError):
    m21 = None
try:
    import miditoolkit  # for CI import
except (ImportError, ModuleNotFoundError):
    miditoolkit = None


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
        >>> tempo, note_list = reader['key1']
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
        self.data = read_2columns_text(fname)  # get key-value dict

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
                if lr is None or lr == "" or lr == "ー":  # multi note in one syllable
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
                    st += dur
                    continue
                else:  # normal note for one syllable
                    notes_list.append(NOTE(lr, note.pitch.midi, st, st + dur))
                prepitch = note.pitch.midi
                for arti in note.articulations:
                    # NOTE(Yuning): By default, 'breath mark' appears at the end of
                    # the sentence. In some situations, 'breath mark' doesn't take
                    # effect in its belonging note. Please handle them under local/.
                    if arti.name in ["breath mark"]:  # <br> is tagged as a notation
                        notes_list.append(NOTE("B", 0, st + dur, st + dur))
                    # NOTE(Yuning): In some datasets, there is a break when 'staccato'
                    # occurs. We let users to decide whether to perform segmentation
                    # under local/.
            else:  # rest note
                if prepitch == 0:
                    notes_list[-1].et += dur
                else:
                    notes_list.append(NOTE("P", 0, st, st + dur))
                prepitch = 0
            st += dur
        # NOTE(Yuning): implicit rest at the end of xml file should be removed.
        if notes_list[-1].midi == 0 and notes_list[-1].lyric == "P":
            notes_list.pop()
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
            if notes_seq[i] != 0:  # isNote
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


class MIDReader(collections.abc.Mapping):
    """Reader class for 'mid.scp'.

    Examples:
        key1 /some/path/a.mid
        key2 /some/path/b.mid
        key3 /some/path/c.mid
        key4 /some/path/d.mid
        ...

        >>> reader = XMLScpReader('mid.scp')
        >>> tempo, note_list = reader['key1']
    """

    def __init__(
        self,
        fname,
        add_rest=True,
        dtype=np.int16,
    ):
        assert check_argument_types()
        assert miditoolkit is not None, (
            "Cannot load miditoolkit package. ",
            "Please install Muskit modules via ",
            "(cd tools && make muskit.done)",
        )
        self.fname = fname
        self.dtype = dtype
        self.add_rest = add_rest  # add rest into note sequencee
        self.data = read_2columns_text(fname)  # get key-value dict

    def __getitem__(self, key):
        midi_obj = miditoolkit.midi.parser.MidiFile(self.data[key])
        # load tempo
        tempos = midi_obj.tempo_changes
        tempos.sort(key=lambda x: (x.time, x.tempo))
        assert len(tempos) == 1
        tempo = int(tempos[0].tempo + 0.5)
        # load pitch time sequence
        tick_to_time = midi_obj.get_tick_to_time_mapping()
        notes = midi_obj.instruments[0].notes
        notes.sort(key=lambda x: (x.start, x.pitch))
        notes_list = []
        pre_et = 0
        for note in notes:
            st = tick_to_time[note.start]
            et = tick_to_time[note.end]
            # NOTE(Yuning): MIDIs don't have explicit rest notes.
            # Explicit rest notes might be needed for stage 1 in svs.
            if st != pre_et and self.add_rest:
                notes_list.append(NOTE("P", 0, pre_et, st))
            notes_list.append(NOTE("*", note.pitch, st, et))
            pre_et = et
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
        self.data = read_2columns_text(fname)

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
