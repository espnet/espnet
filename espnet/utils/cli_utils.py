import io
import logging
import os
import sys

import h5py
import kaldiio
import numpy
import soundfile

from espnet.utils.io_utils import SoundHDF5File

PY2 = sys.version_info[0] == 2

if PY2:
    from collections import Sequence
else:
    # The ABCs from 'collections' will stop working in 3.8
    from collections.abc import Sequence


def get_commandline_args():
    extra_chars = [' ', ';', '&', '(', ')', '|', '^', '<', '>', '?', '*',
                   '[', ']', '$', '`', '"', '\\', '!', '{', '}']

    # Escape the extra characters for shell
    argv = [arg.replace('\'', '\'\\\'\'')
            if all(char not in arg for char in extra_chars)
            else '\'' + arg.replace('\'', '\'\\\'\'') + '\''
            for arg in sys.argv]

    return sys.executable + ' ' + ' '.join(argv)


def is_scipy_wav_style(value):
    # If Tuple[int, numpy.ndarray] or not
    return (isinstance(value, Sequence) and len(value) == 2 and
            isinstance(value[0], int) and
            isinstance(value[1], numpy.ndarray))


def assert_scipy_wav_style(value):
    assert is_scipy_wav_style(value), \
        'Must be Tuple[int, numpy.ndarray], but got {}'.format(
            type(value) if not isinstance(value, Sequence)
            else '{}[{}]'.format(type(value),
                                 ', '.join(str(type(v)) for v in value)))


class FileReaderWrapper(object):
    """Read uttid and array in kaldi style

    :param str rspecifier: Give as "ark:feats.ark" or "scp:feats.scp"
    :param str filetype: "mat" is kaldi-martix, "hdf5": HDF5
    :param bool return_shape: Return the shape of the matrix,
        instead of the matrix. This can reduce IO cost for HDF5.
    :rtype: Generator[Tuple[str, np.ndarray], None, None]

    Read from kaldi-matrix ark file:

    >>> for u, array in FileReaderWrapper('ark:feats.ark', 'mat'):
    ...     array

    Read from HDF5 file:

    >>> for u, array in FileReaderWrapper('ark:feats.h5', 'hdf5'):
    ...     array

    This might be a bit confusing as "ark" is used for HDF5 to imitate kaldi.

    """

    def __init__(self, rspecifier, filetype='mat', return_shape=False,
                 segments=None):
        if segments is None and filetype != 'mat':
            raise ValueError('Not supporting segments if filetype={}'
                             .format(filetype))

        self.rspecifier = rspecifier
        self.filetype = filetype
        self.return_shape = return_shape
        self.segments = segments

    def __iter__(self):
        if self.filetype == 'mat':
            with kaldiio.ReadHelper(
                    self.rspecifier, segments=self.segments) as reader:
                for key, array in reader:
                    if self.return_shape:
                        array = array.shape
                    yield key, array

        elif self.filetype == 'sound':
            if ':' not in self.rspecifier:
                raise ValueError('Give "rspecifier" such as "scp:some.scp: {}"'
                                 .format(self.rspecifier))
            ark_or_scp, filepath = self.rspecifier.split(':', 1)
            if ark_or_scp != 'scp':
                raise ValueError('Only supporting "scp" for sound file: {}'
                                 .format(ark_or_scp))
            with io.open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    key, sound_file_path = line.rstrip().split(None, 1)
                    # Assume PCM16
                    array, rate = soundfile.read(sound_file_path,
                                                 dtype='int16')
                    # Change Tuple[ndarray, int] -> Tuple[int, ndarray]
                    # (soundfile style -> scipy style)

                    if self.return_shape:
                        array = array.shape
                    yield key, (rate, array)

        elif self.filetype in ['hdf5', 'sound.hdf5']:
            if ':' not in self.rspecifier:
                raise ValueError('Give "rspecifier" such as "ark:some.ark: {}"'
                                 .format(self.rspecifier))
            ark_or_scp, filepath = self.rspecifier.split(':', 1)
            if ark_or_scp not in ['ark', 'scp']:
                raise ValueError('Must be scp or ark: {}'.format(ark_or_scp))

            if ark_or_scp == 'scp':
                hdf5_dict = {}
                with io.open(filepath, 'r', encoding='utf-8') as f:
                    for line in f:
                        key, value = line.rstrip().split(None, 1)

                        if ':' not in value:
                            raise RuntimeError(
                                'scp file for hdf5 should be like: '
                                '"uttid filepath.h5:key": {}({})'
                                .format(line, filepath))
                        path, h5_key = value.split(':', 1)

                        hdf5_file = hdf5_dict.get(path)
                        if hdf5_file is None:
                            try:
                                if self.filetype == 'sound.hdf5':
                                    hdf5_file = SoundHDF5File(path, 'r')
                                else:
                                    hdf5_file = h5py.File(path, 'r')
                            except Exception:
                                logging.error(
                                    'Error when loading {}'.format(path))
                                raise
                            hdf5_dict[path] = hdf5_file

                        try:
                            data = hdf5_file[h5_key]
                        except Exception:
                            logging.error('Error when loading {} with key={}'
                                          .format(path, h5_key))
                            raise

                        if self.filetype == 'sound.hdf5':
                            # Change Tuple[ndarray, int] -> Tuple[int, ndarray]
                            # (soundfile style -> scipy style)
                            array, rate = data

                            if self.return_shape:
                                array = array.shape
                            yield key, (rate, array)
                        else:
                            if self.return_shape:
                                yield key, data.shape
                            else:
                                yield key, data[()]

                # Closing all files
                for k in hdf5_dict:
                    hdf5_dict[k].close()

            else:
                if filepath == '-':
                    # Required h5py>=2.9
                    if PY2:
                        filepath = io.BytesIO(sys.stdin.read())
                    else:
                        filepath = io.BytesIO(sys.stdin.buffer.read())
                if self.filetype == 'sound.hdf5':
                    for key, (r, a) in SoundHDF5File(filepath, 'r').items():
                        if self.return_shape:
                            a = a.shape
                        yield key, (r, a)
                else:
                    with h5py.File(filepath, 'r') as f:
                        for key in f:
                            if self.return_shape:
                                yield key, f[key].shape
                            else:
                                yield key, f[key][()]
        else:
            raise ValueError(
                'Not supporting: filetype={}'.format(self.filetype))


class FileWriterWrapper(object):
    """Write matrices in kaldi style

    :param str wspecifier:
    :param str filetype: "mat" is kaldi-martix, "hdf5": HDF5
    :param str write_num_frames: e.g. 'ark,t:num_frames.txt'
    :param bool compress: Compress or not
    :param int compression_method: Specify compression level

    Write in kaldi-matrix-ark with "kaldi-scp" file:

    >>> with FileWriterWrapper('ark,scp:out.ark,out.scp') as f:
    >>>     f['uttid'] = array

    This "scp" has the following format:

        uttidA out.ark:1234
        uttidB out.ark:2222

    where, 1234 and 2222 points the strating byte address of the matrix.
    (For detail, see official documentation of Kaldi)

    Write in HDF5 with "scp" file:

    >>> with FileWriterWrapper('ark,scp:out.h5,out.scp', 'hdf5') as f:
    >>>     f['uttid'] = array

    This "scp" file is created as:

        uttidA out.h5:uttidA
        uttidB out.h5:uttidB

    HDF5 can be, unlike "kaldi-ark", accessed to any keys,
    so originally "scp" is not required for random-reading.
    Nevertheless we create "scp" for HDF5 because it is useful
    for some use-case. e.g. Concatenation, Splitting.

    """

    def __init__(self, wspecifier, filetype='mat',
                 write_num_frames=None, compress=False, compression_method=2,
                 pcm_format='wav'):
        self.writer_scp = None
        # Used for writing scp
        self.filename = None
        self.filetype = filetype
        # Used for filetype='sound' or 'sound.hdf5'
        self.pcm_format = pcm_format
        self.kwargs = {}

        if filetype == 'mat':
            if compress:
                self.writer = kaldiio.WriteHelper(
                    wspecifier, compression_method=compression_method)
            else:
                self.writer = kaldiio.WriteHelper(wspecifier)

        elif filetype in ['hdf5', 'sound.hdf5', 'sound']:
            # 1. Create spec_dict

            # e.g.
            #   ark,scp:out.ark,out.scp -> {'ark': 'out.ark', 'scp': 'out.scp'}
            ark_scp, filepath = wspecifier.split(':', 1)
            if ark_scp not in ['ark', 'scp,ark', 'ark,scp']:
                raise ValueError(
                    '{} is not allowed: {}'.format(ark_scp, wspecifier))
            ark_scps = ark_scp.split(',')
            filepaths = filepath.split(',')
            if len(ark_scps) != len(filepaths):
                raise ValueError(
                    'Mismatch: {} and {}'.format(ark_scp, filepath))
            spec_dict = dict(zip(ark_scps, filepaths))

            # 2. Set writer
            self.filename = spec_dict['ark']

            if filetype == 'sound.hdf5':
                self.writer = SoundHDF5File(spec_dict['ark'], 'w',
                                            format=self.pcm_format)

            elif filetype == 'hdf5':
                self.writer = h5py.File(spec_dict['ark'], 'w')

            elif filetype == 'sound':
                # Use "ark" value as directory to save wav files
                # e.g. ark,scp:dirname,wav.scp
                # -> The wave files are found in dirname/*.wav
                wavdir = spec_dict['ark']
                if not os.path.exists(wavdir):
                    os.makedirs(wavdir)
                self.writer = None
            else:
                # Cannot reach
                raise RuntimeError

            # 3. Set writer_scp
            if 'scp' in spec_dict:
                self.writer_scp = io.open(
                    spec_dict['scp'], 'w', encoding='utf-8')

        else:
            raise ValueError('Not supporting: filetype={}'.format(filetype))

        if write_num_frames is not None:
            if ':' not in write_num_frames:
                raise ValueError('Must include ":", write_num_frames={}'
                                 .format(write_num_frames))

            nframes_type, nframes_file = write_num_frames.split(':', 1)
            if nframes_type != 'ark,t':
                raise ValueError(
                    'Only supporting text mode. '
                    'e.g. --write-num-frames=ark,t:foo.txt :'
                    '{}'.format(nframes_type))

            self.writer_nframe = io.open(nframes_file, 'w', encoding='utf-8')
        else:
            self.writer_nframe = None

    def __setitem__(self, key, value):
        if self.filetype == 'mat':
            self.writer[key] = value

        elif self.filetype == 'hdf5':
            self.writer.create_dataset(key, data=value, **self.kwargs)

        elif self.filetype == 'sound.hdf5':
            assert_scipy_wav_style(value)
            # Change Tuple[int, ndarray] -> Tuple[ndarray, int]
            # (scipy style -> soundfile style)
            value = (value[1], value[0])
            self.writer.create_dataset(key, data=value, **self.kwargs)

        elif self.filetype == 'sound':
            assert_scipy_wav_style(value)
            rate, signal = value

            wavfile = os.path.join(self.filename, key + '.' + self.pcm_format)
            soundfile.write(wavfile, signal.astype(numpy.int16), rate)

        else:
            # Cannot reach
            raise NotImplementedError

        if self.writer_scp is not None:
            if self.filetype in ['hdf5', 'sound.hdf5']:
                self.writer_scp.write(
                    u'{} {}:{}\n'.format(key, self.filename, key))
            elif self.filetype in 'sound':
                wavfile = os.path.join(
                    self.filename, key + '.' + self.pcm_format)
                self.writer_scp.write(u'{} {}\n'.format(key, wavfile))
            else:
                # Cannot reach
                raise NotImplementedError

        if self.writer_nframe is not None:
            self.writer_nframe.write(u'{} {}\n'.format(key, len(value)))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        try:
            self.writer.close()
        except Exception:
            pass

        if self.writer_scp is not None:
            try:
                self.writer_scp.close()
            except Exception:
                pass

        if self.writer_nframe is not None:
            try:
                self.writer_nframe.close()
            except Exception:
                pass
