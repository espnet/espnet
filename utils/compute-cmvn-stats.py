#!/usr/bin/env python

#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import logging

import h5py
import kaldi_io_py
import numpy as np


def main():
    parser = argparse.ArgumentParser(
        description='Compute cepstral mean and '
                    'variance normalization statistics'
                    'If wspecifier provided: per-utterance by default, '
                    'or per-speaker if'
                    'spk2utt option provided; if wxfilename: global')
    parser.add_argument('--spk2utt', type=str,
                        help='A text file of speaker to utterance-list map. '
                             '(Don\'t give rspecifier format, such as '
                             '"ark:utt2spk")')
    parser.add_argument('rspecifier', type=str,
                        help='Read specifier for feats. e.g. ark:some.ark')
    parser.add_argument('wspecifier_or_wxfilename', type=str,
                        help='Output file id. e.g. ark:some.ark or some.ark')
    args = parser.parse_args()

    # logging info
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=logfmt)

    if ':' not in args.rspecifier:
        raise RuntimeError('Give "rspecifier" such as "ark:some.ark: {}"'
                           .format(args.rspecifier))
    ftype, filepath = args.rspecifier.split(':', 1)
    if ftype == 'scp':
        matrices = kaldi_io_py.read_mat_scp(filepath)
    elif ftype == 'ark':
        matrices = kaldi_io_py.read_mat_ark(filepath)
    elif ftype == 'h5':
        matrices = h5py.File(filepath).items()
    else:
        raise RuntimeError('The file type must be one of scp,ark, or hdf5: {}'
                           .format(ftype))

    is_wspecifier = ':' in args.wspecifier_or_wxfilename

    if is_wspecifier:
        if args.spk2utt is not None:
            utt2spk_dict = {}
            with open(args.spk2utt) as f:
                for line in f:
                    spk, utts = line.split(None, 1)
                    for utt in utts.split():
                        utt2spk_dict[utt] = spk
            # Speaker cmvn
            utt2spk = lambda x: utt2spk_dict[x]
        else:
            # Per utterance cmvn
            utt2spk = lambda x: x
    else:
        # Global cmvn
        if args.spk2utt is not None:
            logging.warning('spk2utt is not used when wxfilename is specified')
        utt2spk = lambda x: None

    # Calculate stats for each speaker
    counts = {}
    sum_feats = {}
    square_sum_feats = {}

    for utt, matrix in matrices:
        assert isinstance(matrix, np.ndarray), type(matrix)
        assert matrix.ndim == 2, matrix.ndim
        spk = utt2spk(utt)

        # Init at the first seen of the spk
        if spk not in counts:
            counts[spk] = 0
            feat_dim = matrix.shape[1]
            # Accumulate in double precision
            sum_feats[spk] = np.zeros((feat_dim,), dtype=np.float64)
            square_sum_feats[spk] = np.zeros((feat_dim,), dtype=np.float64)

        counts[spk] += matrix.shape[0]
        sum_feats[spk] += matrix.sum(axis=0)
        square_sum_feats[spk] += (matrix ** 2).sum(axis=0)

    cmvn_stats = {}
    for spk in counts:
        feat_dim = len(sum_feats[spk])
        _cmvn_stats = np.empty((2, feat_dim + 1), dtype=np.float64)
        _cmvn_stats[0, :-1] = sum_feats[spk]
        _cmvn_stats[1, :-1] = square_sum_feats[spk]
        _cmvn_stats[0, -1] = counts[spk]
        _cmvn_stats[1, -1] = 0.

        cmvn_stats[spk] = _cmvn_stats

    if is_wspecifier:
        ftype, filepath = args.wspecifier_or_wxfilename.split(':', 1)
        if ftype in ['h5,scp', 'scp,h5', 'h5']:
            if ftype == 'h5,scp':
                h5_file, scp_file = filepath.split(',')
            elif ftype == 'scp,h5':
                scp_file, h5_file = filepath.split(',')
            elif ftype == 'h5':
                h5_file = filepath
                scp_file = None
            else:
                # Can't reach
                raise RuntimeError()
            if scp_file is not None:
                fscp = open(scp_file, 'w')
            else:
                fscp = None

            with h5py.File(h5_file, 'w') as f:
                for spk, mat in cmvn_stats.items():
                    f[spk] = mat
                    if fscp is not None:
                        fscp.write('{} {}:{}\n'.format(spk, h5_file, spk))
            if fscp is not None:
                fscp.close()

        else:
            # Use an external command: "copy-feats"
            # FIXME(kamo): copy-feats change the precision to float?
            arkscp = 'ark:| copy-feats --print-args=false ark:- {}'.format(
                args.wspecifier_or_wxfilename)
            with kaldi_io_py.open_or_fd(arkscp, 'wb') as f:
                for spk, mat in cmvn_stats.items():
                    kaldi_io_py.write_mat(f, mat, spk)
    else:
        kaldi_io_py.write_mat(args.wspecifier_or_wxfilename, cmvn_stats[None])


if __name__ == "__main__":
    main()
