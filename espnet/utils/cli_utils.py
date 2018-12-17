import sys

import h5py


def get_commandline_args():
    extra_chars = [' ', ';', '&', '(', ')', '|', '^', '<', '>', '?', '*',
                   '[', ']', '$', '`', '"', '\\', '!', '{', '}']

    argv = [arg.replace('\'', '\'\\\'\'')
            if all(char not in arg for char in extra_chars)
            else '\'' + arg.replace('\'', '\'\\\'\'') + '\''
            for arg in sys.argv]

    return sys.executable + ' ' + ' '.join(argv)


def read_hdf5_scp(filepath):
    hdf5_dict = {}
    with open(filepath) as f:
        for line in f:
            key, value = line.rstrip().split(None, 1)
            if ':' not in value:
                raise RuntimeError(
                    'scp file for hdf5 should have such format: '
                    '"uttid filepath.h5:key": {}({})'.format(line, filepath))
            path, h5_key = value.split(':', 1)
            hdf5_file = hdf5_dict.get(path)
            if hdf5_file is None:
                hdf5_file = h5py.File(path, 'r')
                hdf5_dict[path] = hdf5_file
            yield key, hdf5_file[h5_key].value

