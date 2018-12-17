#!/usr/bin/env python

import argparse
from distutils.util import strtobool
import logging
import sys

import h5py
import kaldi_io_py

from espnet.utils.cli_utils import get_commandline_args
from espnet.utils.cli_utils import read_hdf5_scp

PY2 = sys.version_info[0] == 2


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('--in-filetype', type=str, default='mat',
                        choices=['mat', 'hdf5'],
                        help='Specify the file format for the rspecifier. '
                             '"mat" is the matrix format in kaldi')
    parser.add_argument('--out-filetype', type=str, default='mat',
                        choices=['mat', 'hdf5'],
                        help='Specify the file format for the wspecifier. '
                             '"mat" is the matrix format in kaldi')
    parser.add_argument('--write-num-frames', type=str,
                        help='Specify wspecifer for utt2num_frames')
    parser.add_argument('--compress', type=strtobool, default=False,
                        help='Save in compressed format')
    parser.add_argument('--compression-method', type=int, default=2,
                        help='Specify the method(if mat) or gzip-level(if hdf5)')
    parser.add_argument('rspecifier', type=str,
                        help='Read specifier for feats. e.g. ark:some.ark')
    parser.add_argument('wspecifier', type=str,
                        help='Output file id. e.g. ark:some.ark or some.ark')
    args = parser.parse_args()

    # logging info
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
        if filepath == '-':
            if PY2:
                filepath = sys.stdin
            else:
                filepath = sys.stdin.buffer

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

    ftype, filepath = args.wspecifier.split(':', 1)
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
        if args.write_num_frames is not None:
            nframes_type, nframes_file = args.write_num_frames.split(':')
            if nframes_type != 'ark,t':
                raise NotImplementedError(
                    'Only supporting --write-num-frames=ark,t:foo.txt :'
                    '{}'.format(nframes_type))
            fnframes = open(nframes_file, 'w')
        else:
            fnframes = None

        with h5py.File(h5_file, 'w') as f:
            for utt, mat in matrices:
                if args.compress:
                    kwargs = dict(compression='gzip',
                                  compression_opts=args.compression_method)
                else:
                    kwargs = {}
                f.create_dataset(utt, data=mat, **kwargs)
                if fscp is not None:
                    fscp.write('{} {}:{}\n'.format(utt, h5_file, utt))
                if fnframes is not None:
                    fnframes.write('{} {}\n'.format(utt, len(mat)))
        if fscp is not None:
            fscp.close()
        if fnframes is not None:
            fnframes.close()

    elif args.out_filetype == 'mat':
        # Use an external command: "copy-feats"
        # FIXME(kamo): copy-feats change the precision to float?
        arkscp = 'ark:| copy-feats --print-args=false ark:- {}'.format(
            args.wspecifier)
        if args.compress:
            arkscp = arkscp.replace(
                'copy-feats',
                'copy-feats --compress={} --compression-method={}'
                .format(args.compress, args.compression_method))
        if args.write_num_frames is not None:
            arkscp = arkscp.replace(
                'copy-feats',
                'copy-feats {}'.format(args.write_num_frames))

        with kaldi_io_py.open_or_fd(arkscp, 'wb') as f:
            for spk, mat in matrices:
                kaldi_io_py.write_mat(f, mat, spk)
    else:
        raise RuntimeError('Not supporting: --out-filetype {}'
                           .format(args.out_filetype))


if __name__ == "__main__":
    main()
