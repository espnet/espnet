#!/usr/bin/python
# -*- coding: utf-8 -*-

import argparse
import os
import subprocess
import sys

from utils import json2trn
from utils import trn2ctm
from utils import trn2stm

is_python2 = sys.version_info[0] == 2


def main(args):
    parser = argparse.ArgumentParser()
    parser.add_argument('json', type=str, default=None, nargs='?',
                        help='input trn')
    parser.add_argument('dict', type=str, help='dict')
    parser.add_argument('--num-spkrs', type=int, default=1, nargs='?', help='number of speakers')
    parser.add_argument('--refs', type=str, nargs='*', help='ref for all speakers')
    parser.add_argument('--hyps', type=str, nargs='*', help='hyp for all outputs')
    parser.add_argument('--orig-stm', type=str, nargs='?', help='orig stm')
    parser.add_argument('--stm', type=str, default=None, nargs='+', help='output stm')
    parser.add_argument('--ctm', type=str, default=None, nargs='+', help='output ctm')
    parser.add_argument('--bpe', type=str, default=None, nargs='?', help='BPE model if applicable')
    args = parser.parse_args(args)
    if args.refs is None:
        refs = ["ref_tmp.trn"]
        del_ref = True
    else:
        refs = args.refs
        del_ref = False
    if args.hyps is None:
        hyps = ["hyp_tmp.trn"]
        del_hyp = True
    else:
        hyps = args.hyps
        del_hyp = False
    json2trn.convert(args.json, args.dict, refs, hyps, args.num_spkrs)
    for trn in refs + hyps:
        # We don't remove non-lang-syms because kaldi already removes them when scoring
        call_args = ["sed", "-i.bak2", "-r", "s/<blank> //g", trn]
        subprocess.check_call(call_args)
        if args.bpe is not None:
            with open(wrd_name(trn), 'w') as out:
                with open(trn, 'r') as spm_in:
                    sed_args = ["sed", "-e", "s/▁/ /g"]
                    sed = subprocess.Popen(sed_args, stdout=out, stdin=subprocess.PIPE)
                    spm_args = ["spm_decode", "--model=" + args.bpe, "--input_format=piece"]
                    subprocess.Popen(spm_args, stdin=spm_in)
                    sed.communicate()
        else:
            call_args = ["sed", "-e", "s/ //g", "-e", "s/(/ (/", "-e", "s/<space>/ /g", trn]
            with open(wrd_name(trn), 'w') as out:
                sed = subprocess.Popen(call_args, stdout=out)
                sed.communicate()
    for trn, stm in zip(refs, args.stm):
        trn2stm.convert(wrd_name(trn), stm, args.orig_stm)
    if del_ref:
        os.remove(refs[0])
        os.remove(refs[0] + ".bak2")
        os.remove(wrd_name(refs[0]))

    for trn, ctm in zip(hyps, args.ctm):
        trn2ctm.convert(wrd_name(trn), ctm)
    if del_hyp:
        os.remove(hyps[0])
        os.remove(hyps[0] + ".bak2")
        os.remove(wrd_name(hyps[0]))


def wrd_name(trn):
    split = trn.split(".")
    return ".".join(split[:-1]) + ".wrd." + split[-1]


if __name__ == "__main__":
    main(sys.argv[1:])
