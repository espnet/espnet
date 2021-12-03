#!/usr/bin/env python3
import argparse
import logging
import os
import sys

from espnet.utils.cli_utils import get_commandline_args


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Combine source and target datasets",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--log_level",
        type=lambda x: x.upper(),
        default="INFO",
        choices=("CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG", "NOTSET"),
        help="The verbose level of logging",
    )

    parser.add_argument(
        "--srcspk", required=True, help="Source speaker"
    )
    parser.add_argument(
        "--trgspk", required=True, help="Target speaker"
    )
    parser.add_argument(
        "--src_dir", required=True, help="Source dir"
    )
    parser.add_argument(
        "--trg_dir", required=True, help="Target dir"
    )

    return parser


def main(cmd=None):
    print(get_commandline_args(), file=sys.stderr)
    parser = get_parser()
    args = parser.parse_args(cmd)
    pair_spk = '%s_%s' % (args.srcspk, args.trgspk)
    src_fs = os.listdir(args.src_dir)
    src_fs = [f for f in src_fs if os.path.isfile(os.path.join(args.src_dir, f)) \
                and not f.startswith('.') and not f.endswith('.npz')]
    src_fs = sorted(src_fs)
    trg_fs = os.listdir(args.trg_dir)
    trg_fs = [f for f in trg_fs if os.path.isfile(os.path.join(args.trg_dir, f)) \
                and not f.startswith('.') and not f.endswith('.npz')]
    trg_fs = sorted(trg_fs)
    assert len(src_fs) == len(trg_fs)
    for src_f, trg_f in zip(src_fs, trg_fs):
        src_p = os.path.join(args.src_dir, src_f)
        trg_p = os.path.join(args.trg_dir, trg_f)
        with open(src_p, 'r') as inf:
            lines = inf.readlines()
        src_lines = [l.strip() for l in lines]
        with open(trg_p, 'r') as inf:
            lines = inf.readlines()
        trg_lines = [l.strip() for l in lines]
        assert len(src_lines) == len(trg_lines)
        src_elem0 = src_lines[0].split()[0]
        if src_elem0.startswith(args.srcspk):
            if not src_elem0.startswith(pair_spk):
                new_src_lines = []
                new_trg_lines = []
                if src_elem0 == args.srcspk:
                    assert len(src_lines) == 1  # assumes single speaker
                    assert len(trg_lines) == 1
                    new_fids = [pair_spk+fid[len(args.srcspk):] for fid in src_lines[0].split()[1:]]
                    new_fids_str = ' '.join(new_fids)
                    new_src_line = "%s %s" % (pair_spk, new_fids_str)
                    new_src_lines.append(new_src_line)
                    new_fids = [pair_spk+fid[len(args.trgspk):] for fid in trg_lines[0].split()[1:]]
                    new_fids_str = ' '.join(new_fids)
                    new_trg_line = "%s %s" % (pair_spk, new_fids_str)
                    new_trg_lines.append(new_trg_line)
                else:
                    src_elem1 = src_lines[0].split()[1]
                    if src_elem1 == args.srcspk:
                        for l in src_lines:
                            new_src_lines.append(pair_spk+l.split()[0][len(args.srcspk):]+' '+pair_spk)
                        for l in trg_lines:
                            new_trg_lines.append(pair_spk+l.split()[0][len(args.trgspk):]+' '+pair_spk)
                    else:
                        for l in src_lines:
                            new_src_lines.append(pair_spk+l[len(args.srcspk):])
                        for l in trg_lines:
                            new_trg_lines.append(pair_spk+l[len(args.trgspk):])
                with open(src_p, 'w+') as ouf:
                    for l in new_src_lines:
                        ouf.write('%s\n' % l)
                with open(trg_p, 'w+') as ouf:
                    for l in new_trg_lines:
                        ouf.write('%s\n' % l)


if __name__ == "__main__":
    main()
