import collections.abc
import logging

import numpy as np
from typeguard import check_argument_types

from espnet2.fileio.read_text import read_2column_text

try:
    import music21 as m21  # for CI import
except ImportError or ModuleNotFoundError:
    m21 = None


class XMLScpReader(collections.abc.Mapping):
    # TODO(Yuning): It is designed for sub xmls after segmentation.

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
        part = score.parts[0].flat
        notes_list = []
        prepitch = None
        for note in part.notesAndRests:
            if note.isNote:
                if note.lyric is not None:
                    notes_list.append(note)
                else:
                    if note.pitch == prepitch:
                        notes_list[-1].seconds += note.seconds
                    else:
                        # TODO(Yuning): If there is more than one pitch in a syllable.
                        # Doesn't happen in Ofuton
                        logging.error("More than one pitch in one syllable")
                prepitch = note.pitch
            else:
                break
        m = score.metronomeMarkBoundaries()
        tempo = m[0][2].number
        seq_len = len(notes_list)
        lyrics = []
        notes = np.zeros(seq_len, dtype=self.dtype)
        segs = np.zeros((seq_len, 2))
        t = 0
        for i in range(seq_len):
            st = t
            et = st + notes_list[i].seconds
            segs[i, 0] = np.float32(st)
            segs[i, 1] = np.float32(et)
            t += notes_list[i].seconds
            notes[i] = notes_list[i].pitch.midi
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


# TODO(Yuning): Add XMLScpWriter
# The present XMLScpReader extract note info and ignore rest_note.
# For XMLWriter, additional information
#                (eg. offset, <br>, instrument, etc) might be required.
