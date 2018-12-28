import io
from io import BytesIO
import logging
import sys

import h5py
import kaldiio


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


def read_rspecifier(rspecifier, filetype='mat'):
    """Yield a pair of the uttid and ndarray

    :param str filepath:
    :param str filetype:
    :rtype: Generator[Tuple[str, np.ndarray], None, None]
    """
    if filetype == 'mat':
        with kaldiio.ReadHelper(rspecifier) as reader:
            for key, array in reader:
                yield key, array

    elif filetype == 'hdf5':
        if ':' not in rspecifier:
            raise ValueError('Give "rspecifier" such as "ark:some.ark: {}"'
                             .format(rspecifier))
        ftype, filepath = rspecifier.split(':', 1)
        if ftype not in ['ark', 'scp']:
            raise ValueError('The scp, ark: {}'.format(ftype))

        if ftype == 'scp':
            hdf5_dict = {}
            with io.open(filepath, 'r', encoding='utf-8') as f:
                for line in f:
                    key, value = line.rstrip().split(None, 1)
                    if ':' not in value:
                        raise RuntimeError(
                            'scp file for hdf5 should have such format: '
                            '"uttid filepath.h5:key": {}({})'
                            .format(line, filepath))
                    path, h5_key = value.split(':', 1)

                    hdf5_file = hdf5_dict.get(path)
                    if hdf5_file is None:
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
                yield key, dataset[...]

    else:
        raise ValueError('Not supporting: filetype={}'.format(filetype))


class FileWriterWrapper(object):
    def __init__(self, wspecifier, filetype='mat',
                 write_num_frames=None, compress=False, compression_method=2):
        self.writer_scp = None
        self.filename = None
        self.filetype = filetype
        self.kwargs = {}

        if filetype == 'mat':
            if compress:
                self.writer = kaldiio.WriteHelper(
                    wspecifier, compression_method=compression_method)
            else:
                self.writer = kaldiio.WriteHelper(wspecifier)

        elif filetype == 'hdf5':
            # ark,scp:out.ark,out.scp -> {'ark': 'out.ark', 'scp': 'out.scp'}
            spec_dict = kaldiio.parse_specifier(wspecifier)

            if 'ark,t' in spec_dict:
                logging.warning('Text mode is not supported for HDF5')
                spec_dict['ark'] = spec_dict['ark,t']
            if 'ark' not in spec_dict:
                raise ValueError('Must specify ark file: e.g. ark:out.ark: {}'
                                 .format(wspecifier))

            self.filename = spec_dict['ark']
            self.writer = h5py.File(spec_dict['ark'], 'w')
            if compress:
                self.kwargs = dict(
                    compression='gzip', compression_opts=compression_method)

            if 'scp' in spec_dict:
                self.writer_scp = io.open(spec_dict['scp'], 'w',
                                          encoding='utf-8')
        else:
            raise ValueError('Not supporting: filetype={}'.format(filetype))

        if write_num_frames is not None:
            if ':' not in write_num_frames:
                raise ValueError('Must include ":", write_num_frames={}'
                                 .format(write_num_frames))

            nframes_type, nframes_file = write_num_frames.split(':')
            if nframes_type != 'ark,t':
                raise NotImplementedError(
                    'Only supporting --write-num-frames=ark,t:foo.txt :'
                    '{}'.format(nframes_type))

            self.writer_nframe = io.open(nframes_file, 'w', encoding='utf-8')
        else:
            self.writer_nframe = None

    def __setitem__(self, key, value):
        if self.filetype == 'mat':
            self.writer[key] = value
        elif self.filetype == 'hdf5':
            self.writer.create_dataset(key, data=value, **self.kwargs)
        else:
            raise NotImplementedError

        if self.writer_scp is not None:
            if self.filetype == 'hdf5':
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
