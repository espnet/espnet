import collections.abc
import logging
from pathlib import Path
from typing import Union
from miditoolkit import midi

import numpy as np
import soundfile
from typeguard import check_argument_types
import miditoolkit

from espnet2.fileio.read_text import read_2column_text
from espnet2.fileio.utils import midi_to_seq, seq_to_midi


class MIDIScpReader(collections.abc.Mapping):
    """Reader class for 'midi.scp'.
    Examples:
        key1 /some/path/a.midi
        key2 /some/path/b.midi
        key3 /some/path/c.midi
        key4 /some/path/d.midi
        ...
        >>> reader = MIDIScpReader('midi.scp')
        >>> pitch_array, tempo_array = reader['key1']
    """

    def __init__(
        self,
        fname,
        dtype=np.int16,
        loader_type: str = "representation",
        rate: np.int32 = np.int32(16000),
        mode: str = "format",
        time_shift: float = 0.0125,
    ):
        assert check_argument_types()
        self.fname = fname
        self.dtype = dtype
        self.rep = loader_type
        self.rate = rate
        self.mode = mode
        self.time_shift = time_shift
        self.data = read_2column_text(fname)  # get key-value dict

    def __getitem__(self, key):
        key, pitch_aug_factor, time_aug_factor = key
        # return miditoolkit.midi.parser.MidiFile(self.data[key])
        midi_obj = miditoolkit.midi.parser.MidiFile(self.data[key])

        if self.rep == "representation":
            note_seq, tempo_seq = midi_to_seq(
                midi_obj,
                self.dtype,
                self.rate,
                pitch_aug_factor,
                time_aug_factor,
                self.mode,
                self.time_shift,
            )
        else:
            raise TypeError("Not supported loader type {}".format(self.rep))
        return note_seq, tempo_seq

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


class MIDIScpWriter:
    """Writer class for 'midi.scp'
    Examples:
        key1 /some/path/a.midi
        key2 /some/path/b.midi
        key3 /some/path/c.midi
        key4 /some/path/d.midi
        ...
        >>> writer = MIDIScpWriter('./data/', './data/midi.scp')
        >>> writer['aa'] = midi_obj
        >>> writer['bb'] = midi_obj
    """

    def __init__(
        self,
        outdir: Union[Path, str],
        scpfile: Union[Path, str],
        format="midi",
        dtype=None,
        rate: np.int32 = np.int32(16000),
    ):
        assert check_argument_types()
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")
        self.rate = rate
        self.format = format
        self.dtype = dtype

        self.data = {}

    def __setitem__(self, key: str, value: tuple):
        assert len(value) == 2, "The midi values should include  both note and tempo"
        note_seq, tempo_seq = value
        midi_path = self.dir / f"{key}.{self.format}"
        midi_path.parent.mkdir(parents=True, exist_ok=True)
        midi_obj = seq_to_midi(note_seq, tempo_seq, self.rate)
        notes = midi_obj.instruments[0].notes

        if len(notes) > 0:
            midi_obj.dump(midi_path)
            self.fscp.write(f"{key} {midi_path}\n")
            # Store the file path
            self.data[key] = str(midi_path)
        else:
            logging.warning(
                f"no corresponding note sequence for segments {key}. skip it"
            )

    def get_path(self, key):
        return self.data[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()


# if __name__ == "__main__":
#     # . /data3/qt/Muskits/tools/activate_python.sh; python /data3/qt/Muskits/muskit/fileio/midi_scp.py
#     from tqdm import tqdm
#     import os
#     args_segments = '/data3/qt/Muskits/egs/kiritan/svs1/data/train/segments'
#     scp = '/data3/qt/Muskits/egs/kiritan/svs1/data/train/midi.scp'
#     fs = np.int16(16000)
#     outdir = '/data3/qt/Muskits/egs/kiritan/test'
#     out_midiscp = outdir+"/test.scp"


#     segments = {}
#     with open(args_segments) as f:
#         for line in f:
#             if len(line) == 0:
#                 continue
#             utt_id, recording_id, segment_begin, segment_end = line.strip().split(
#                 " "
#             )
#             segments[utt_id] = (
#                 recording_id,
#                 float(segment_begin),
#                 float(segment_end),
#             )


#     loader = MIDIScpReader(scp, rate=fs)
#     writer = MIDIScpWriter(
#         outdir,
#         out_midiscp,
#         format="midi",
#         rate=fs,
#     )
#     cache = (None, None, None)
#     for utt_id, (recording, start, end) in tqdm(segments.items()):
#         # TODO: specify track information here
#         if recording == cache[0]:
#             note_seq, tempo_seq = cache[1], cache[2]
#         else:
#             note_seq, tempo_seq = loader[recording]
#             cache = (recording, note_seq, tempo_seq)

#         if fs is not None:
#             start = int(start * fs)
#             end = int(end * fs)
#             if start < 0:
#                 start = 0
#             if end > len(note_seq):
#                 end = len(note_seq)
#         else:
#             start = np.searchsorted([item[0] for item in note_seq], start, "left")
#             end = np.searchsorted([item[1] for item in note_seq], end, "left")
#         sub_note = note_seq[start:end]
#         sub_tempo = tempo_seq[start:end]

#         writer[utt_id] = sub_note, sub_tempo

#     else:
#         # midi_scp does not need to change, when no segments is applied
#         # Note things will change, after finish other todos in the script
#         os.system("cp {} {}".format(scp, Path(outdir / f"{test}.scp")))

#     path = '/data3/qt/Muskits/egs/kiritan/svs1/dump/raw/org/train/data/format_midi.18/kiritan11_0000.midi'
#     midi_obj = miditoolkit.midi.parser.MidiFile(path)
#     note_seq, tempo_seq = midi_to_seq(midi_obj, np.int16, np.int16(16000) )
