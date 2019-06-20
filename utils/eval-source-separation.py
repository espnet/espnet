#!/usr/bin/env python
import argparse
from collections import OrderedDict
from distutils.util import strtobool
import itertools
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
import warnings

import museval
import numpy as np
from pystoi.stoi import stoi
import soundfile

from espnet.utils.cli_utils import get_commandline_args


def eval_STOI(ref, y, fs, extended=False, compute_permutation=True):
    """Calculate STOI

    Reference:
        A short-time objective intelligibility measure
            for time-frequency weighted noisy speech
        https://ieeexplore.ieee.org/document/5495701

    Note(kamo):
        STOI is defined on the signal at 10kHz
        and the input at the other sampling rate will be resampled.
        Thus, the result differs depending on the implementation of resampling.
        Especially, pystoi cannot reproduce matlab's resampling now.

    :param ref (np.ndarray): Reference (Nsrc, Nframe, Nmic)
    :param y (np.ndarray): Enhanced (Nsrc, Nframe, Nmic)
    :param fs (int): Sample frequency
    :param extended (bool): stoi or estoi
    :param compute_permutation (bool):
    :return: value, perm
    :rtype: Tuple[Tuple[float, ...], Tuple[int, ...]]
    """
    if ref.shape != y.shape:
        raise ValueError('ref and y should have the same shape: {} != {}'
                         .format(ref.shape, y.shape))
    if ref.ndim != 3:
        raise ValueError('Input must have 3 dims: {}'.format_map(ref.ndim))
    n_src = ref.shape[0]
    n_mic = ref.shape[2]

    if compute_permutation:
        index_list = list(itertools.permutations(range(n_src)))
    else:
        index_list = [list(range(n_src))]

    values = [[sum(stoi(ref[i, :, ch], y[j, :, ch], fs, extended)
                   for ch in range(n_mic)) / n_mic
               for i, j in enumerate(indices)]
              for indices in index_list]

    best_pairs = sorted([(v, i) for v, i in zip(values, index_list)],
                        key=lambda x: sum(x[0]))[-1]
    value, perm = best_pairs
    return tuple(value), tuple(perm)


def eval_PESQ(ref, enh, fs, compute_permutation: bool = True,
              wideband: bool = True):
    """Evaluate PESQ

    PESQ program can be downloaded from here:
        http://www.itu.int/rec/dologin_pub.asp?lang=e&id=T-REC-P.862-200511-I!Amd2!SOFT-ZST-E&type=items

    Reference:
        Perceptual evaluation of speech quality (PESQ)-a new method
            for speech quality assessment of telephone networks and codecs
        https://ieeexplore.ieee.org/document/941023

    :param x (np.ndarray): Reference (Nsrc, Nframe, Nmic)
    :param y (np.ndarray): Enhanced (Nsrc, Nframe, Nmic)
    :param fs (int): Sample frequency
    :param compute_permutation (bool):
    """
    if shutil.which('PESQ') is None:
        raise RuntimeError('PESQ: command not found: Please install')
    if fs not in (8000, 16000):
        raise ValueError('Sample frequency must be 8000 or 16000: {}'
                         .format(fs))
    if ref.shape != enh.shape:
        raise ValueError('ref and enh should have the same shape: {} != {}'
                         .format(ref.shape, enh.shape))
    if ref.ndim != 3:
        raise ValueError('Input must have 3 dims: {}'.format_map(ref.ndim))

    n_src = ref.shape[0]
    n_mic = ref.shape[2]
    with TemporaryDirectory() as d:
        # Dumping wav files temporary
        ref_files = []
        enh_files = []
        for isrc in range(n_src):
            refs = []  # [Nsrc, Nmic]
            enhs = []  # [Nsrc, Nmic]
            for imic in range(n_mic):
                wv = str(os.path.join(d, 'ref.{}.{}.wav'.format(isrc, imic)))
                soundfile.write(wv, ref[isrc, :, imic].astype(np.int16), fs)
                refs.append(wv)

                wv = str(os.path.join(d, 'enh.{}.{}.wav'.format(isrc, imic)))
                soundfile.write(wv, enh[isrc, :, imic].astype(np.int16), fs)
                enhs.append(wv)
            ref_files.append(refs)
            enh_files.append(enhs)

        if compute_permutation:
            index_list = list(itertools.permutations(range(n_src)))
        else:
            index_list = [list(range(n_src))]

        values = []
        for indices in index_list:
            values2 = []
            for i, j in enumerate(indices):
                lis = []
                for imic in range(n_mic):
                    # PESQ +<8000|16000> <ref.wav> <enh.wav> [smos] [cond]
                    if wideband:
                        commands = ['PESQ', '+{}'.format(fs), '+wb',
                                    ref_files[i][imic], enh_files[j][imic]]
                    else:
                        commands = ['PESQ', '+{}'.format(fs),
                                    ref_files[i][imic], enh_files[j][imic]]
                    with subprocess.Popen(commands, stdout=subprocess.DEVNULL,
                                          cwd=d) as p:
                        _, _ = p.communicate()

                    # e.g.
                    # REFERENCE	 DEGRADED	 PESQMOS	 MOSLQO	 SAMPLE_FREQ	 MODE
                    # /tmp/tmp9k5rxgjd/ref.0.wav	 /tmp/tmp9k5rxgjd/enh.0.wav	 -1.000	 4.644	 16000	wb
                    result_txt = (Path(d) / 'pesq_results.txt')
                    if result_txt.exists():
                        with result_txt.open('r') as f:
                            lis.append(float(f.readlines()[1].split()[3]))
                    else:
                        # Sometimes PESQ is failed. I don't know why.
                        warnings.warn('Processing error is found.')
                        lis.append(1.)
                    # Averaging over n_mic
                # Averaging over n_mic
                values2.append(sum(lis) / len(lis))
            values.append(values2)
    best_pairs = sorted([(v, i) for v, i in zip(values, index_list)],
                        key=lambda x: sum(x[0]))[-1]
    value, perm = best_pairs
    return tuple(value), tuple(perm)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Evaluate enhanced speech. '
                    'e.g. {c} --ref ref.scp --enh enh.scp --outdir outputdir'
                    'or {c} --ref ref.scp ref2.scp --enh enh.scp enh2.scp '
                    '--outdir outputdir'
                    .format(c=sys.argv[0]),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--verbose', '-V', default=0, type=int,
                        help='Verbose option')
    parser.add_argument('--ref', dest='reffiles', nargs='+', type=str,
                        required=True,
                        help='WAV file lists for reference')
    parser.add_argument('--enh', dest='enhfiles', nargs='+', type=str,
                        required=True,
                        help='WAV files lists for enhanced')
    parser.add_argument('--outdir', type=str,
                        required=True)
    parser.add_argument('--keylist', type=str,
                        help='Specify the target samples. By default, '
                             'using all keys in the first reference file')
    parser.add_argument('--evaltypes', type=str, nargs='+',
                        choices=['SDR', 'STOI', 'ESTOI', 'PESQ'],
                        default=['SDR', 'STOI', 'ESTOI', 'PESQ'])
    parser.add_argument('--permutation', type=strtobool, default=True,
                        help='Compute all permutations or '
                             'use the pair of input order')

    # About BSS Eval v4:
    # The 2018 Signal Separation Evaluation Campaign
    # https://arxiv.org/abs/1804.06267
    parser.add_argument('--bss-eval-images', type=strtobool, default=True,
                        help='Use bss_eval_images or bss_eval_sources. '
                             'For more detail, see museval source codes.')
    parser.add_argument('--bss-eval-version', type=str,
                        default='v3', choices=['v3', 'v4'],
                        help='Specify bss-eval-version: v3 or v4')
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    if args.verbose > 0:
        logging.basicConfig(level=logging.INFO, format=logfmt)
    else:
        logging.basicConfig(level=logging.WARN, format=logfmt)
    logging.info(get_commandline_args())
    if len(args.reffiles) != len(args.enhfiles):
        raise RuntimeError(
            'The number of ref files are different '
            'from the enh files: {} != {}'.format(len(args.reffiles),
                                                  len(args.enhfiles)))
    if len(args.enhfiles) == 1:
        args.permutation = False

    # Read text files and created a mapping of key2filepath
    reffiles_dict = OrderedDict()  # Dict[str, Dict[str, str]]
    for ref in args.reffiles:
        d = OrderedDict()
        with open(ref, 'r') as f:
            for line in f:
                key, path = line.split(None, 1)
                d[key] = path.rstrip()
        reffiles_dict[ref] = d

    enhfiles_dict = OrderedDict()  # Dict[str, Dict[str, str]]
    for enh in args.enhfiles:
        d = OrderedDict()
        with open(enh, 'r') as f:
            for line in f:
                key, path = line.split(None, 1)
                d[key] = path.rstrip()
        enhfiles_dict[enh] = d

    if args.keylist is not None:
        with open(args.keylist, 'r') as f:
            keylist = [line.rstrip().split()[0] for line in f]
    else:
        keylist = list(reffiles_dict.values())[0]

    if len(keylist) == 0:
        raise RuntimeError('No keys are found')

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    evaltypes = []
    for evaltype in args.evaltypes:
        if evaltype == 'SDR':
            evaltypes += ['SDR', 'ISR', 'SIR', 'SAR']
        else:
            evaltypes.append(evaltype)

    # Open files in write mode
    writers = {k: open(os.path.join(args.outdir, k), 'w') for k in evaltypes}

    for key in keylist:
        # 1. Load ref files
        rate_prev = None

        ref_signals = []
        for listname, d in reffiles_dict.items():
            if key not in d:
                raise RuntimeError('{} doesn\'t exist in {}'
                                   .format(key, listname))
            filepath = d[key]
            signal, rate = soundfile.read(filepath, dtype=np.int16)
            if signal.ndim == 1:
                # (Nframe) -> (Nframe, 1)
                signal = signal[:, None]
            ref_signals.append(signal)
            if rate_prev is not None and rate != rate_prev:
                raise RuntimeError('Sampling rates mismatch')
            rate_prev = rate

        # 2. Load enh files
        enh_signals = []
        for listname, d in enhfiles_dict.items():
            if key not in d:
                raise RuntimeError('{} doesn\'t exist in {}'
                                   .format(key, listname))
            filepath = d[key]
            signal, rate = soundfile.read(filepath, dtype=np.int16)
            if signal.ndim == 1:
                # (Nframe) -> (Nframe, 1)
                signal = signal[:, None]
            enh_signals.append(signal)
            if rate_prev is not None and rate != rate_prev:
                raise RuntimeError('Sampling rates mismatch')
            rate_prev = rate

        for signal in ref_signals + enh_signals:
            if signal.shape[1] != ref_signals[0].shape[1]:
                raise RuntimeError('The number of channels mismatch')

        # 3. Zero padding to adjust the length to the maximum length in inputs
        ml = max(len(s) for s in ref_signals + enh_signals)
        ref_signals = [np.pad(s, [(0, ml - len(s)), (0, 0)], mode='constant')
                       if len(s) < ml else s for s in ref_signals]

        enh_signals = [np.pad(s, [(0, ml - len(s)), (0, 0)], mode='constant')
                       if len(s) < ml else s for s in enh_signals]

        # ref_signals, enh_signals: (Nsrc, Nframe, Nmic)
        ref_signals = np.stack(ref_signals, axis=0)
        enh_signals = np.stack(enh_signals, axis=0)

        # 4. Evaluates
        for evaltype in args.evaltypes:
            if evaltype == 'SDR':
                (sdr, isr, sir, sar, perm) = \
                    museval.metrics.bss_eval(
                        ref_signals, enh_signals,
                        window=np.inf, hop=np.inf,
                        compute_permutation=args.permutation,
                        filters_len=512,
                        framewise_filters=args.bss_eval_version == 'v3',
                        bsseval_sources_version=not args.bss_eval_images)

                # sdr: (Nsrc, Nframe)
                writers['SDR'].write(
                    '{} {}\n'.format(key, ' '.join(map(str, sdr[:, 0]))))
                writers['ISR'].write(
                    '{} {}\n'.format(key, ' '.join(map(str, isr[:, 0]))))
                writers['SIR'].write(
                    '{} {}\n'.format(key, ' '.join(map(str, sir[:, 0]))))
                writers['SAR'].write(
                    '{} {}\n'.format(key, ' '.join(map(str, sar[:, 0]))))

            elif evaltype == 'STOI':
                stoi, perm = eval_STOI(ref_signals, enh_signals, rate,
                                       extended=False,
                                       compute_permutation=args.permutation)
                writers['STOI'].write(
                    '{} {}\n'.format(key, ' '.join(map(str, stoi))))

            elif evaltype == 'ESTOI':
                estoi, perm = eval_STOI(ref_signals, enh_signals, rate,
                                        extended=True,
                                        compute_permutation=args.permutation)
                writers['ESTOI'].write(
                    '{} {}\n'.format(key, ' '.join(map(str, estoi))))

            elif evaltype == 'PESQ':
                pesq, perm = eval_PESQ(ref_signals, enh_signals, rate,
                                       compute_permutation=args.permutation)
                writers['PESQ'].write(
                    '{} {}\n'.format(key, ' '.join(map(str, pesq))))
            else:
                # Cannot reach
                raise RuntimeError


if __name__ == "__main__":
    main()
