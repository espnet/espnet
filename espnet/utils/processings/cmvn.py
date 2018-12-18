import io

import h5py
import kaldi_io_py
import numpy as np


class CMVN(object):
    def __init__(self, stats, norm_means=True, norm_vars=False,
                 filetype='mat', utt2spk=None, spk2utt=None,
                 std_floor=1.0e-20):
        self.stats_file = stats
        self.norm_means = norm_means
        self.norm_vars = norm_vars

        # Use for global CMVN
        if filetype == 'mat':
            stats_dict = {None: kaldi_io_py.read_mat(stats)}
        # Use for global CMVN
        elif filetype == 'npy':
            stats_dict = {None: np.load(stats)}
        # Use for speaker CMVN
        elif filetype == 'ark':
            stats_dict = dict(kaldi_io_py.read_mat_ark(stats))
        # Use for speaker CMVN
        elif filetype == 'hdf5':
            stats_dict = h5py.File(stats)
        else:
            raise ValueError('Not supporting filetype={}'.format(filetype))

        if utt2spk is not None:
            self.utt2spk = {}
            with io.open(utt2spk, 'r', encoding='utf-8') as f:
                for line in f:
                    utt, spk = line.rstrip().split(None, 1)
                    self.utt2spk[utt] = spk
        elif spk2utt is not None:
            self.utt2spk = {}
            with io.open(spk2utt, 'r', encoding='utf-8') as f:
                for line in f:
                    spk, utts = line.rstrip().split(None, 1)
                    for utt in utts.split():
                        self.utt2spk[utt] = spk
        else:
            self.utt2spk = None

        # Kaldi makes a matrix for CMVN which has a shape of (2, feat_dim + 1),
        # and the first vector contains the sum of feats and the second is
        # the sum of squares. The last value of the first, i.e. stats[0,1],
        # is the number of samples for this statistics.
        self.bias = {}
        self.scale = {}
        for spk, stats in stats_dict.items():
            assert len(stats) == 2, stats.shape

            count = stats[0, -1]

            # If the feature has two or more dimensions
            if not (np.isscalar(count) or isinstance(count, (int, float))):
                # The first is only used
                count = count.flattten()[0]

            mean = stats[0, :-1] / count
            # V(x) = E(x^2) - (E(x))^2
            var = stats[1, :-1] / count - mean * mean
            std = np.maximum(np.sqrt(var), std_floor)
            self.bias[spk] = -mean
            self.scale[spk] = 1 / std

    def __repr__(self):
        return ('{name}(stats_file={stats_file}, '
                'norm_means={norm_means}, norm_vars={norm_vars})'
                .format(name=self.__class__.__name__,
                        stats_file=self.stats_file,
                        norm_means=self.norm_means,
                        norm_vars=self.norm_vars))

    def __call__(self, x, uttid=None):
        if self.utt2spk is not None:
            spk = self.utt2spk[uttid]
        else:
            spk = uttid

        if self.norm_means:
            np.add(x, self.bias[spk], x, dtype=x.dtype)
        if self.norm_vars:
            np.multiply(x, self.scale[spk], x, dtype=x.dtype)
        return x
