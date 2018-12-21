import sys
from io import BytesIO

import h5py
import kaldi_io_py


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
    if ':' not in rspecifier:
        raise ValueError('Give "rspecifier" such as "ark:some.ark: {}"'
                         .format(rspecifier))
    ftype, filepath = rspecifier.split(':', 1)
    if ftype not in ['ark', 'scp']:
        raise ValueError('The scp, ark: {}'.format(ftype))

    if filetype == 'mat':
        if filepath == '-':
            if PY2:
                filepath = sys.stdin
            else:
                filepath = sys.stdin.buffer

        if ftype == 'scp':
            matrices = kaldi_io_py.read_mat_scp(filepath)
        else:
            matrices = kaldi_io_py.read_mat_ark(filepath)
        for key, array in matrices:
            yield key, array

    elif filetype == 'hdf5':
        if ftype == 'scp':
            hdf5_dict = {}
            with open(filepath) as f:
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
                    yield key, hdf5_file[h5_key].value

        else:
            if filepath == '-':
                # Required h5py>=2.9
                if PY2:
                    filepath = BytesIO(sys.stdin.read())
                else:
                    filepath = BytesIO(sys.stdin.buffer.read())
            for key, dataset in h5py.File(filepath, 'r').items():
                yield key, dataset.value

    else:
        raise NotImplementedError(
            'Not supporting: filetype={}'.format(filetype))
