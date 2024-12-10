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
    """
    Read audio files using the soundfile library.

    This function reads one or more audio files and returns their audio data
    as a NumPy array along with the sample rate. It can handle both single
    and multi-channel audio files, and allows for optional concatenation of
    audio data along a specified axis.

    Args:
        wavs (Union[str, List[str]]): A single audio file path or a list of
            audio file paths to read.
        dtype: The desired data type of the output array (default is None).
        always_2d (bool, optional): If True, ensures that the output array
            is always 2-dimensional (default is False).
        concat_axis (int, optional): The axis along which to concatenate
            multiple audio arrays (default is 1).
        start (int, optional): The starting frame to read from each audio
            file (default is 0).
        end (int, optional): The ending frame to read from each audio file.
            If None, reads until the end of the file (default is None).
        return_subtype (bool, optional): If True, returns the subtype of
            the audio files along with the data and sample rate (default is
            False).

    Returns:
        Tuple[np.array, int]: A tuple containing the audio data as a
        NumPy array and the sample rate of the audio files. If
        return_subtype is True, it returns a tuple of the form
        (array, rate, subtypes).

    Raises:
        RuntimeError: If the sampling rates of the audio files do not
        match or if the shapes of the arrays do not align along the
        specified concatenation axis.

    Examples:
        >>> data, rate = soundfile_read('audio.wav')
        >>> data, rate = soundfile_read(['audio1.wav', 'audio2.wav'],
        ...                              concat_axis=0)
        >>> data, rate, subtypes = soundfile_read('audio.wav',
        ...                                        return_subtype=True)

    Note:
        This function uses the soundfile library, which needs to be
        installed separately. Ensure that the audio files are in a
        supported format.
    """
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
    """
        Reader class for 'wav.scp'.

    This class reads a 'wav.scp' file which contains mappings of keys to audio
    file paths. It can handle both single and multi-column entries for audio
    files. The multi-column option allows for concatenating multiple audio
    files associated with a single key.

    Attributes:
        fname (str): The path to the 'wav.scp' file.
        dtype (optional): Data type for the audio data.
        always_2d (bool): If True, ensures the output is always 2-dimensional.
        multi_columns (bool): If True, enables reading of multi-column entries.
        concat_axis (int): Axis along which to concatenate audio data.

    Args:
        fname (str): The path to the 'wav.scp' file.
        dtype (optional): Data type for audio data (default: None).
        always_2d (bool): Ensure output is always 2D (default: False).
        multi_columns (bool): Enable reading of multi-column entries (default: False).
        concat_axis (int): Axis for concatenation of audio data (default: 1).

    Examples:
        wav.scp is a text file that looks like the following:

            key1 /some/path/a.wav
            key2 /some/path/b.wav
            key3 /some/path/c.wav
            key4 /some/path/d.wav

        >>> reader = SoundScpReader('wav.scp')
        >>> rate, array = reader['key1']

        If multi_columns=True is given and multiple files are given in one line
        with space delimiter, the output array is concatenated along the
        channel direction:

            key1 /some/path/a.wav /some/path/a2.wav
            key2 /some/path/b.wav /some/path/b2.wav

        >>> reader = SoundScpReader('wav.scp', multi_columns=True)
        >>> rate, array = reader['key1']

        In the above case, a.wav and a2.wav are concatenated.

        Note that even if multi_columns=True is given, SoundScpReader still
        supports a normal wav.scp, i.e., a wav file is given per line,
        but this option is disabled by default because a dict[str, list[str]]
        object is needed to be kept, which increases the required amount of memory.
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
        """
            Retrieve the file path associated with a given key.

        This method accesses the internal data structure of the
        SoundScpReader class to return the file path corresponding
        to the specified key.

        Args:
            key (str): The key for which to retrieve the associated
                file path.

        Returns:
            Union[str, List[str]]: The file path or list of file paths
                associated with the key. If the key corresponds to
                multiple files, a list of paths is returned.

        Raises:
            KeyError: If the key is not found in the data.

        Examples:
            >>> reader = SoundScpReader('wav.scp')
            >>> path = reader.get_path('key1')
            >>> print(path)
            /some/path/a.wav

            >>> reader_multi = SoundScpReader('multi_wav.scp', multi_columns=True)
            >>> paths = reader_multi.get_path('key1')
            >>> print(paths)
            ['/some/path/a.wav', '/some/path/a2.wav']
        """
        return self.data[key]

    def __contains__(self, item):
        return item

    def __len__(self):
        return len(self.data)

    def __iter__(self):
        return iter(self.data)

    def keys(self):
        """
            Reader class for 'wav.scp'.

        This class provides an interface to read audio file paths from a
        'wav.scp' file, which maps keys to their corresponding audio file
        paths. It supports both single and multi-column formats for
        handling audio files.

        Attributes:
            fname (str): The path to the 'wav.scp' file.
            dtype (Optional[str]): The data type for audio reading.
            always_2d (bool): If True, ensures that the returned audio arrays
                are always 2D.
            multi_columns (bool): If True, allows multiple audio files to be
                specified in one line, separated by spaces.
            concat_axis (int): The axis along which to concatenate audio arrays
                if multi-columns is used.

        Args:
            fname (str): The path to the 'wav.scp' file.
            dtype (Optional[str]): The data type for audio reading.
            always_2d (bool): If True, ensures that the returned audio arrays
                are always 2D. Default is False.
            multi_columns (bool): If True, allows multiple audio files to be
                specified in one line, separated by spaces. Default is False.
            concat_axis (int): The axis along which to concatenate audio arrays
                if multi-columns is used. Default is 1.

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
            with space delimiter, the output array will be concatenated
            along the channel direction:

            key1 /some/path/a.wav /some/path/a2.wav
            key2 /some/path/b.wav /some/path/b2.wav
            ...

            >>> reader = SoundScpReader('wav.scp', multi_columns=True)
            >>> rate, array = reader['key1']

            In the above case, a.wav and a2.wav are concatenated.

            Note that even if multi_columns=True is given,
            SoundScpReader still supports a normal wav.scp,
            i.e., a wav file is given per line,
            but this option is disabled by default
            because a dict[str, list[str]] object is needed to be kept,
            which increases the required amount of memory.
        """
        return self.data.keys()


class SoundScpWriter:
    """
        Writer class for 'wav.scp'.

    This class allows for writing audio file paths and their corresponding
    sampling rates to a specified 'wav.scp' file. It supports writing
    single-channel or multi-channel audio data, as well as specifying
    output formats and naming conventions for the generated audio files.

    Attributes:
        dir (Path): Directory where audio files will be saved.
        fscp (TextIOWrapper): File handle for writing to the .scp file.
        format (str): The output audio format (default is 'wav').
        subtype (Optional[str]): Subtype for the audio files (e.g., 'PCM_16').
        output_name_format (str): Naming format for generated audio files.
        multi_columns (bool): If True, saves multi-channel data as multiple
            monaural audio files.
        output_name_format_multi_columns (str): Naming format for generated
            audio files when multi_columns is enabled.
        data (dict): Dictionary storing the mapping of keys to file paths.

    Args:
        outdir (Union[Path, str]): Directory where audio files will be saved.
        scpfile (Union[Path, str]): Path to the output 'wav.scp' file.
        format (str): The output audio format (default is 'wav').
        multi_columns (bool): Save multi-channel data as multiple monaural
            audio files (default is False).
        output_name_format (str): The naming format of generated audio files
            (default is '{key}.{audio_format}').
        output_name_format_multi_columns (str): The naming format of generated
            audio files when multi_columns is given (default is
            '{key}-CH{channel}.{audio_format}').
        dtype (Optional): Data type of the audio signal (default is None).
        subtype (Optional[str]): Subtype for the audio files (default is None).

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

    Note:
        The audio files will be written in the specified output directory.
        Ensure that the directory exists or is created before writing.
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
        """
            Retrieve the file path associated with the given key.

        Args:
            key (str): The key for which to retrieve the associated file path.

        Returns:
            Union[str, List[str]]: The file path or list of file paths
            corresponding to the specified key in the wav.scp format.

        Examples:
            >>> writer = SoundScpWriter('./data/', './data/wav.scp')
            >>> writer['sample_key'] = 16000, numpy_array
            >>> writer.get_path('sample_key')
            './data/sample_key.wav'

            >>> writer = SoundScpWriter('./data/', './data/wav.scp',
            ... multi_columns=True)
            >>> writer['multi_key'] = 16000, numpy_array
            >>> writer.get_path('multi_key')
            ['./data/multi_key-CH0.wav', './data/multi_key-CH1.wav']
        """
        return self.data[key]

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """
        Writer class for 'wav.scp'.

            This class allows for writing audio file paths to a 'wav.scp' file, where
            each line corresponds to a key and its associated audio file. It can handle
            both single-channel and multi-channel audio data, with the ability to specify
            the output format and naming conventions for generated audio files.

            Args:
                outdir: The output directory where audio files will be saved.
                scpfile: The path to the 'wav.scp' file that will be created.
                format: The output audio format (default: "wav").
                multi_columns: If True, saves multi-channel data as multiple monaural
                    audio files (default: False).
                output_name_format: The naming format of generated audio files (default:
                    "{key}.{audio_format}").
                output_name_format_multi_columns: The naming format of generated audio
                    files when multi_columns is True (default: "{key}-CH{channel}.{audio_format}").
                dtype: Data type of the audio file (optional).
                subtype: Subtype of the audio file (optional).

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

            Attributes:
                dir: The output directory as a Path object.
                fscp: The file handle for the 'wav.scp' file.
                format: The audio format for the output files.
                subtype: The audio subtype, if specified.
                output_name_format: The naming format for output files.
                multi_columns: Boolean indicating if multi-channel files should be saved.
                output_name_format_multi_columns: The naming format for multi-channel files.
                data: A dictionary storing the paths of written audio files.

            Raises:
                ValueError: If the value tuple does not contain exactly two elements.
                TypeError: If the value is not a tuple of int and numpy.ndarray.
                RuntimeError: If the input signal is not 1 or 2 dimensional.
        """
        self.fscp.close()
