# Copyright 2017 Shigeki Karita
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)
from kaldi_io_py import open_or_fd, read_mat


class ScpLazyDict(object):
    def __init__(self, loader_dict):
        self.loader_dict = loader_dict

    def __getitem__(self, item):
        return read_mat(self.loader_dict[item.decode('utf-8')])


def read_dict_scp(file_or_fd):
    """ ScpLazyDict = read_mat_scp(file_or_fd)
    Returns LazyScpDict with __getitem__ to read kaldi ark according to kaldi scp.
    file_or_fd : scp, gzipped scp, pipe or opened file descriptor.
    """
    fd = open_or_fd(file_or_fd)
    d = dict()
    try:
        for line in fd:
            key, rxfile = line.decode('utf-8').split(' ')
            d[key] = rxfile
    finally:
        if fd is not file_or_fd:
            fd.close()
    return ScpLazyDict(d)
