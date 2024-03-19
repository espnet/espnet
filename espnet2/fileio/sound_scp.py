import collections.abc
from pathlib import Path
from typing import List, Optional, Tuple, Union

import numpy as np
import soundfile
from typeguard import typechecked

from espnet2.fileio.read_text import read_2columns_text, read_multi_columns_text


def soundfile_read(
    wavs: Union[str, List[str]],
    dtype=None,
    always_2d: bool = False,
    concat_axis: int = 1,
    start: int = 0,
    end: int = None,
    return_subtype: bool = False,
) -> Tuple[np.array, int]:
    if isinstance(wavs, str):
        wavs = [wavs]

    arrays = []
    subtypes = []
    prev_rate = None
    prev_wav = None
    for wav in wavs:
        with soundfile.SoundFile(wav) as f:
            f.seek(start)
            if end is not None:
                frames = end - start
            else:
                frames = -1
            if dtype == "float16":
                array = f.read(
                    frames,
                    dtype="float32",
                    always_2d=always_2d,
                ).astype(dtype)
            else:
                array = f.read(frames, dtype=dtype, always_2d=always_2d)
            rate = f.samplerate
            subtype = f.subtype
            subtypes.append(subtype)

        if len(wavs) > 1 and array.ndim == 1 and concat_axis == 1:
            # array: (Time, Channel)
            array = array[:, None]

        if prev_wav is not None:
            if prev_rate != rate:
                raise RuntimeError(
                    f"'{prev_wav}' and '{wav}' have mismatched sampling rate: "
                    f"{prev_rate} != {rate}"
                )

            dim1 = arrays[0].shape[1 - concat_axis]
            dim2 = array.shape[1 - concat_axis]
            if dim1 != dim2:
                raise RuntimeError(
                    "Shapes must match with "
                    f"{1 - concat_axis} axis, but gut {dim1} and {dim2}"
                )

        prev_rate = rate
        prev_wav = wav
        arrays.append(array)

    if len(arrays) == 1:
        array = arrays[0]
    else:
        array = np.concatenate(arrays, axis=concat_axis)

    if return_subtype:
        return array, rate, subtypes
    else:
        return array, rate


class SoundScpReader(collections.abc.Mapping):
    """Reader class for 'wav.scp'.

    Examples:
        wav.scp is a text file that looks like the following:

        key1 /some/path/a.wav
        key2 /some/path/b.wav
        key3 /some/path/c.wav
        key4 /some/path/d.wav
        ...

        >>> reader = SoundScpReader('wav.scp')
        >>> rate, array = reader['key1']

        If multi_columns=True is given and
        multiple files are given in one line
        with space delimiter, and  the output array are concatenated
        along channel direction

        key1 /some/path/a.wav /some/path/a2.wav
        key2 /some/path/b.wav /some/path/b2.wav
        ...

        >>> reader = SoundScpReader('wav.scp', multi_columns=True)
        >>> rate, array = reader['key1']

        In the above case, a.wav and a2.wav are concatenated.

        Note that even if multi_columns=True is given,
        SoundScpReader still supports a normal wav.scp,
        i.e., a wav file is given per line,
        but this option is disable by default
        because dict[str, list[str]] object is needed to be kept,
        but it increases the required amount of memory.
    """

    @typechecked
    def __init__(
        self,
        fname,
        dtype=None,
        always_2d: bool = False,
        multi_columns: bool = False,
        concat_axis=1,
    ):
        self.fname = fname
        self.dtype = dtype
        self.always_2d = always_2d

        if multi_columns:
            self.data, _ = read_multi_columns_text(fname)
        else:
            self.data = read_2columns_text(fname)
        self.multi_columns = multi_columns
        self.concat_axis = concat_axis

    def __getitem__(self, key) -> Tuple[int, np.ndarray]:
        wavs = self.data[key]

        array, rate = soundfile_read(
            wavs,
            dtype=self.dtype,
            always_2d=self.always_2d,
            concat_axis=self.concat_axis,
        )
        # Returned as scipy.io.wavread's order
        return rate, array

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


class SoundScpWriter:
    """Writer class for 'wav.scp'

    Args:
        outdir:
        scpfile:
        format: The output audio format
        multi_columns: Save multi channel data
            as multiple monaural audio files
        output_name_format: The naming formam of generated audio files
        output_name_format_multi_columns: The naming formam of generated audio files
            when multi_columns is given
        dtype:
        subtype:

    Examples:
        >>> writer = SoundScpWriter('./data/', './data/wav.scp')
        >>> writer['aa'] = 16000, numpy_array
        >>> writer['bb'] = 16000, numpy_array

        aa ./data/aa.wav
        bb ./data/bb.wav

        >>> writer = SoundScpWriter(
            './data/', './data/feat.scp', multi_columns=True,
        )
        >>> numpy_array.shape
        (100, 2)
        >>> writer['aa'] = 16000, numpy_array

        aa ./data/aa-CH0.wav ./data/aa-CH1.wav

    """

    @typechecked
    def __init__(
        self,
        outdir: Union[Path, str],
        scpfile: Union[Path, str],
        format="wav",
        multi_columns: bool = False,
        output_name_format: str = "{key}.{audio_format}",
        output_name_format_multi_columns: str = "{key}-CH{channel}.{audio_format}",
        subtype: Optional[str] = None,
    ):
        self.dir = Path(outdir)
        self.dir.mkdir(parents=True, exist_ok=True)
        scpfile = Path(scpfile)
        scpfile.parent.mkdir(parents=True, exist_ok=True)
        self.fscp = scpfile.open("w", encoding="utf-8")
        self.format = format
        self.subtype = subtype
        self.output_name_format = output_name_format
        self.multi_columns = multi_columns
        self.output_name_format_multi_columns = output_name_format_multi_columns

        self.data = {}

    def __setitem__(
        self, key: str, value: Union[Tuple[int, np.ndarray], Tuple[np.ndarray, int]]
    ):
        value = list(value)
        if len(value) != 2:
            raise ValueError(f"Expecting 2 elements, but got {len(value)}")
        if isinstance(value[0], int) and isinstance(value[1], np.ndarray):
            rate, signal = value
        elif isinstance(value[1], int) and isinstance(value[0], np.ndarray):
            signal, rate = value
        else:
            raise TypeError("value shoulbe be a tuple of int and numpy.ndarray")

        if signal.ndim not in (1, 2):
            raise RuntimeError(f"Input signal must be 1 or 2 dimension: {signal.ndim}")
        if signal.ndim == 1:
            signal = signal[:, None]

        if signal.shape[1] > 1 and self.multi_columns:
            wavs = []
            for channel in range(signal.shape[1]):
                wav = self.dir / self.output_name_format_multi_columns.format(
                    key=key, audio_format=self.format, channel=channel
                )
                wav.parent.mkdir(parents=True, exist_ok=True)
                wav = str(wav)
                soundfile.write(wav, signal[:, channel], rate, subtype=self.subtype)
                wavs.append(wav)

            self.fscp.write(f"{key} {' '.join(wavs)}\n")

            # Store the file path
            self.data[key] = wavs
        else:
            wav = self.dir / self.output_name_format.format(
                key=key, audio_format=self.format
            )
            wav.parent.mkdir(parents=True, exist_ok=True)
            wav = str(wav)
            soundfile.write(wav, signal, rate, subtype=self.subtype)
            self.fscp.write(f"{key} {wav}\n")

            # Store the file path
            self.data[key] = wav

    def get_path(self, key):
        return self.data[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self.fscp.close()
