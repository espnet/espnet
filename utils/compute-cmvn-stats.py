#!/usr/bin/env python

import argparse
import logging

import h5py
import kaldi_io_py
import numpy as np

from espnet.utils.cli_utils import read_hdf5_scp
from espnet.utils.cli_utils import get_commandline_args


def main():
    parser = argparse.ArgumentParser(
        description='Compute cepstral mean and '
                    'variance normalization statistics'
                    'If wspecifier provided: per-utterance by default, '
                    'or per-speaker if'
                    'spk2utt option provided; if wxfilename: global',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--spk2utt', type=str,
                        help='A text file of speaker to utterance-list map. '
                             '(Don\'t give rspecifier format, such as '
                             '"ark:utt2spk")')
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('--in-filetype', type=str, default='mat',
                        choices=['mat', 'hdf5'],
                        help='Specify the file format for the rspecifier. '
                             '"mat" is the matrix format in kaldi')
    parser.add_argument('--out-filetype', type=str, default='mat',
                        choices=['mat', 'hdf5', 'npy'],
                        help='Specify the file format for the wspecifier. '
                             '"mat" is the matrix format in kaldi')
    parser.add_argument('rspecifier', type=str,
                        help='Read specifier for feats. e.g. ark:some.ark')
    parser.add_argument('wspecifier_or_wxfilename', type=str,
                        help='Output file id. e.g. ark:some.ark or some.ark')
    args = parser.parse_args()

    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())

    if ':' not in args.rspecifier:
        raise RuntimeError('Give "rspecifier" such as "ark:some.ark: {}"'
                           .format(args.rspecifier))
    ftype, filepath = args.rspecifier.split(':', 1)
    if ftype not in ['ark', 'scp']:
        raise RuntimeError('The file type must be one of scp, ark: {}'
                           .format(ftype))
    if args.in_filetype == 'mat':
        if ftype == 'scp':
            matrices = kaldi_io_py.read_mat_scp(filepath)
        else:
            matrices = kaldi_io_py.read_mat_ark(filepath)

    elif args.in_filetype == 'hdf5':
        if ftype == 'scp':
            matrices = read_hdf5_scp(filepath)
        else:
            matrices = h5py.File(filepath).items()
    else:
        raise NotImplementedError(
            'Not supporting: --filetype {}'.format(args.filetype))

    is_wspecifier = ':' in args.wspecifier_or_wxfilename

    if is_wspecifier:
        if args.spk2utt is not None:
            logging.info('Performing as speaker CMVN mode')
            utt2spk_dict = {}
            with open(args.spk2utt) as f:
                for line in f:
                    spk, utts = line.rstrip().split(None, 1)
                    for utt in utts.split():
                        utt2spk_dict[utt] = spk

            def utt2spk(x): return utt2spk_dict[x]
        else:
            logging.info('Performing as utterance CMVN mode')

            def utt2spk(x): return x

        if args.out_filetype == 'npy':
            logging.warning('--out-filetype npy is allowed only for '
                            'Global CMVN mode, changing to hdf5')
            args.out_filetype = 'hdf5'

    else:
        logging.info('Performing as global CMVN mode')
        if args.spk2utt is not None:
            logging.warning('spk2utt is not used for global CMVN mode')

        def utt2spk(x): return None

        if args.out_filetype == 'hdf5':
            logging.warning('--out-filetype hdf5 is not allowed for '
                            'Global CMVN mode, changing to npy')
            args.out_filetype = 'npy'

    # Calculate stats for each speaker
    counts = {}
    sum_feats = {}
    square_sum_feats = {}

    idx = 0
    for idx, (utt, matrix) in enumerate(matrices, 1):
        assert isinstance(matrix, np.ndarray), type(matrix)
        spk = utt2spk(utt)

        # Init at the first seen of the spk
        if spk not in counts:
            counts[spk] = 0
            feat_shape = matrix.shape[1:]
            # Accumulate in double precision
            sum_feats[spk] = np.zeros(feat_shape, dtype=np.float64)
            square_sum_feats[spk] = np.zeros(feat_shape, dtype=np.float64)

        counts[spk] += matrix.shape[0]
        sum_feats[spk] += matrix.sum(axis=0)
        square_sum_feats[spk] += (matrix ** 2).sum(axis=0)
    logging.info('Processed {} utterances'.format(idx))
    assert idx > 0, idx

    cmvn_stats = {}
    for spk in counts:
        feat_shape = sum_feats[spk].shape
        cmvn_shape = (2, feat_shape[0] + 1) + feat_shape[1:]
        _cmvn_stats = np.empty(cmvn_shape, dtype=np.float64)
        _cmvn_stats[0, :-1] = sum_feats[spk]
        _cmvn_stats[1, :-1] = square_sum_feats[spk]

        _cmvn_stats[0, -1] = counts[spk]
        _cmvn_stats[1, -1] = 0.

        # You can get the mean and std as following,
        # >>> N = _cmvn_stats[0, 0]
        # >>> mean = _cmvn_stats[0, :-1] / N
        # >>> std = np.sqrt(_cmvn_stats[1, :-1] / N - mean ** 2)

        cmvn_stats[spk] = _cmvn_stats

    # Per utterance or speaker CMVN
    if is_wspecifier:
        ftype, filepath = args.wspecifier_or_wxfilename.split(':', 1)
        if args.out_filetype == 'hdf5':
            if ftype == 'ark,scp':
                h5_file, scp_file = filepath.split(',')
            elif ftype == 'scp,ark':
                scp_file, h5_file = filepath.split(',')
            elif ftype == 'ark':
                h5_file = filepath
                scp_file = None
            else:
                raise RuntimeError(
                    'Give "wspecifier" such as "ark:some.ark: {}"')
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

        elif args.out_filetype == 'mat':
            # Use an external command: "copy-feats"
            # FIXME(kamo): copy-feats change the precision to float?
            arkscp = 'ark:| copy-feats --print-args=false ark:- {}'.format(
                args.wspecifier_or_wxfilename)
            with kaldi_io_py.open_or_fd(arkscp, 'wb') as f:
                for spk, mat in cmvn_stats.items():
                    kaldi_io_py.write_mat(f, mat, spk)
        else:
            raise RuntimeError('Not supporting: --out-filetype {}'
                               .format(args.out_filetype))

    # Global CMVN
    else:
        matrix = cmvn_stats[None]
        if args.out_filetype == 'npy':
            np.save(args.wspecifier_or_wxfilename, matrix)
        elif args.out_filetype == 'mat':
            # Kaldi supports only matrix or vector
            kaldi_io_py.write_mat(args.wspecifier_or_wxfilename, matrix)
        else:
            raise RuntimeError('Not supporting: --out-filetype {}'
                               .format(args.out_filetype))


if __name__ == "__main__":
    main()
