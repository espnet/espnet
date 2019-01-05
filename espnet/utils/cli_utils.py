import io
from io import BytesIO
import sys

import h5py
import kaldiio

from espnet.utils.io_utils import SoundHDF5File

PY2 = sys.version_info[0] == 2


def get_commandline_args():
    extra_chars = [' ', ';', '&', '(', ')', '|', '^', '<', '>', '?', '*',
                   '[', ']', '$', '`', '"', '\\', '!', '{', '}']

    # Escape the extra characters for shell
    argv = [arg.replace('\'', '\'\\\'\'')
            if all(char not in arg for char in extra_chars)
            else '\'' + arg.replace('\'', '\'\\\'\'') + '\''
            for arg in sys.argv]

    return sys.executable + ' ' + ' '.join(argv)


class FileReaderWrapper(object):
    """Yield a pair of the uttid and ndarray

    >>> for u, array in FileReaderWrapper('ark:feats.ark', filetype='mat'):
    ...     array

    :param str rspecifier:
    :param str filetype: "mat" is kaldi-martix, "hdf5": HDF5
    :rtype: Generator[Tuple[str, np.ndarray], None, None]
    """

    def __init__(self, rspecifier, filetype='mat'):
        self.rspecifier = rspecifier
        self.filetype = filetype
        self.keys = set()

    def __contains__(self, item):
        return item in self.keys

    def __iter__(self):
        if self.filetype == 'mat':
            with kaldiio.ReadHelper(self.rspecifier) as reader:
                for key, array in reader:
                    self.keys.add(key)
                    yield key, array

        elif self.filetype in ['hdf5', 'flac.hdf5']:
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
                        self.keys.add(key)

                        if ':' not in value:
                            raise RuntimeError(
                                'scp file for hdf5 should be like: '
                                '"uttid filepath.h5:key": {}({})'
                                .format(line, filepath))
                        path, h5_key = value.split(':', 1)

                        hdf5_file = hdf5_dict.get(path)
                        if hdf5_file is None:
                            if self.filetype == 'flac.hdf5':
                                hdf5_file = SoundHDF5File(path, 'r',
                                                          format='flac')
                            else:
                                hdf5_file = h5py.File(path, 'r')
                            hdf5_dict[path] = hdf5_file
                        yield key, hdf5_file[h5_key][...]

            else:
                if filepath == '-':
                    # Required h5py>=2.9
                    if PY2:
                        filepath = BytesIO(sys.stdin.read())
                    else:
                        filepath = BytesIO(sys.stdin.buffer.read())
                for key, dataset in h5py.File(filepath, 'r').items():
                    self.keys.add(key)
                    yield key, dataset[...]

        else:
            raise ValueError(
                'Not supporting: filetype={}'.format(self.filetype))


class FileWriterWrapper(object):
    """Write matrices in matrix-ark of hdf5 with scp file

    >>> with FileWriterWrapper('ark,scp:out.ark,out.scp') as f:
    >>>     f['uttid'] = array

    :param str wspecifier:
    :param str filetype: "mat" is kaldi-martix, "hdf5": HDF5
    :param str write_num_frames: e.g. 'ark,t:num_frames.txt'
    :param bool compress: Compress or not
    :param int compression_method: Specify compression level

    """

    def __init__(self, wspecifier, filetype='mat',
                 write_num_frames=None, compress=False, compression_method=2):
        self.writer_scp = None
        self.filename = None
        self.filetype = filetype
        self.kwargs = {}
        self.keys = set()

        if filetype == 'mat':
            if compress:
                self.writer = kaldiio.WriteHelper(
                    wspecifier, compression_method=compression_method)
            else:
                self.writer = kaldiio.WriteHelper(wspecifier)

        elif filetype == 'hdf5':
            # ark,scp:out.ark,out.scp -> {'ark': 'out.ark', 'scp': 'out.scp'}
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
            self.writer = h5py.File(spec_dict['ark'])
            if 'scp' in spec_dict:
                self.writer_scp = io.open(
                    spec_dict['scp'], 'w', encoding='utf-8')

        else:
            raise ValueError('Not supporting: filetype={}'.format(filetype))

        if write_num_frames is not None:
            if ':' not in write_num_frames:
                raise ValueError('Must include ":", write_num_frames={}'
                                 .format(write_num_frames))

            nframes_type, nframes_file = write_num_frames.split(':')
            if nframes_type != 'ark,t':
                raise ValueError(
                    'Only supporting text mode. '
                    'e.g. --write-num-frames=ark,t:foo.txt :'
                    '{}'.format(nframes_type))

            self.writer_nframe = io.open(nframes_file, 'w', encoding='utf-8')
        else:
            self.writer_nframe = None

    def __setitem__(self, key, value):
        self.keys.add(key)

        if self.filetype == 'mat':
            self.writer[key] = value
        elif self.filetype in ['hdf5', 'flac.hdf5']:
            self.writer.create_dataset(key, data=value, **self.kwargs)
        else:
            raise NotImplementedError

        if self.writer_scp is not None:
            if self.filetype in ['hdf5', 'flac.hdf5']:
                self.writer_scp.write(
                    '{} {}:{}\n'.format(key, self.filename, key))
            else:
                raise NotImplementedError

        if self.writer_nframe is not None:
            self.writer_nframe.write('{} {}\n'.format(key, len(value)))

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __contains__(self, item):
        return item in self.keys

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
