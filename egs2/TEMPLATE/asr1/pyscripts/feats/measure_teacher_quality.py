# The measure_teacher_quality.py uses code from Fairseq:
#     https://github.com/pytorch/fairseq/blob/master/examples/hubert/measure_teacher_quality.py
#
# Thanks to Abdelrahman Mohamed and Wei-Ning Hsu's help in this implementation,
# Their origial Hubert work is in:
#     Paper: https://arxiv.org/pdf/2106.07447.pdf
#     Code in Fairseq: https://github.com/pytorch/fairseq/tree/master/examples/hubert

import argparse
import re
from collections import Counter

import numpy as np
from tabulate import tabulate

from espnet2.utils.types import str2bool


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lab_dir", type=str, required=True)
    parser.add_argument("--lab_name", type=str, required=True)
    parser.add_argument("--lab_sets", default=["valid"], type=str, nargs="+")
    parser.add_argument(
        "--phn_dir",
        default="data/librispeech_phoneme_alignment",
    )
    parser.add_argument(
        "--phn_sets", default=["dev-clean", "dev-other"], type=str, nargs="+"
    )
    parser.add_argument("--pad_len", default=0, type=int, help="padding for hypotheses")
    parser.add_argument(
        "--upsample", default=1, type=int, help="upsample factor for hypotheses"
    )
    parser.add_argument("--ref_lab_dir", default="")
    parser.add_argument("--ref_lab_name", default="")
    parser.add_argument("--verbose", type=str2bool, default=False)
    return parser


def comp_purity(p_xy, axis):
    max_p = p_xy.max(axis=axis)
    marg_p = p_xy.sum(axis=axis)
    indv_pur = max_p / marg_p
    aggr_pur = max_p.sum()
    return indv_pur, aggr_pur


def comp_entropy(p):
    return (-p * np.log(p + 1e-8)).sum()


def comp_norm_mutual_info(p_xy):
    p_x = p_xy.sum(axis=1, keepdims=True)
    p_y = p_xy.sum(axis=0, keepdims=True)
    pmi = np.log(p_xy / np.matmul(p_x, p_y) + 1e-8)
    mi = (p_xy * pmi).sum()
    h_x = comp_entropy(p_x)
    h_y = comp_entropy(p_y)
    return mi, mi / h_x, mi / h_y, h_x, h_y


def pad(labs, n):
    if n == 0:
        return np.array(labs)
    return np.concatenate([[labs[0]] * n, labs, [labs[-1]] * n])


def comp_avg_seg_dur(labs_list):
    n_frms = 0
    n_segs = 0
    for labs in labs_list:
        labs = np.array(labs)
        edges = np.zeros(len(labs)).astype(bool)
        edges[0] = True
        edges[1:] = labs[1:] != labs[:-1]
        n_frms += len(edges)
        n_segs += edges.astype(int).sum()
    return n_frms / n_segs


def comp_joint_prob(uid2refs, uid2hyps):
    """
    Args:
        pad: padding for spliced-feature derived labels
    """
    cnts = Counter()
    skipped = []
    abs_frmdiff = 0
    for uid in uid2refs:
        if uid not in uid2hyps:
            skipped.append(uid)
            continue
        refs = uid2refs[uid]
        hyps = uid2hyps[uid]

        abs_frmdiff += abs(len(refs) - len(hyps))
        min_len = min(len(refs), len(hyps))
        refs = refs[:min_len]
        hyps = hyps[:min_len]
        cnts.update(zip(refs, hyps))
    tot = sum(cnts.values())

    ref_set = sorted({ref for ref, _ in cnts.keys()})
    hyp_set = sorted({hyp for _, hyp in cnts.keys()})
    ref2pid = dict(zip(ref_set, range(len(ref_set))))
    hyp2lid = dict(zip(hyp_set, range(len(hyp_set))))
    # print(hyp_set)
    p_xy = np.zeros((len(ref2pid), len(hyp2lid)), dtype=float)
    for (ref, hyp), cnt in cnts.items():
        p_xy[ref2pid[ref], hyp2lid[hyp]] = cnt
    p_xy /= p_xy.sum()
    return p_xy, ref2pid, hyp2lid, tot, abs_frmdiff, skipped


def read_phn(tsv_path, rm_stress=True):
    uid2phns = {}
    with open(tsv_path) as f:
        for line in f:
            uid, phns = line.rstrip().split("\t")
            phns = phns.split(",")
            if rm_stress:
                phns = [re.sub("[0-9]", "", phn) for phn in phns]
            uid2phns[uid] = phns
    return uid2phns


def read_lab(lab_path, pad_len=0, upsample=1):
    with open(lab_path) as f:
        labs_list = [
            (
                line.rstrip().split()[0],
                pad(line.rstrip().split()[1:], pad_len).repeat(upsample),
            )
            for line in f
        ]
    return dict(labs_list)


def main_lab_lab(
    lab_dir,
    lab_name,
    lab_sets,
    ref_dir,
    ref_name,
    pad_len=0,
    upsample=1,
    verbose=False,
):
    uid2refs = {}
    for s in lab_sets:
        uid2refs.update(read_lab(f"{ref_dir}/{s}.{ref_name}"))

    uid2hyps = {}
    for s in lab_sets:
        uid2hyps.update(
            read_lab(
                f"{lab_dir}/{s}.{lab_name}",
                pad_len,
                upsample,
            )
        )
    _main(uid2refs, uid2hyps, verbose)


def main_phn_lab(
    lab_dir,
    lab_name,
    lab_sets,
    phn_dir,
    phn_sets,
    pad_len=0,
    upsample=1,
    verbose=False,
):
    uid2refs = {}
    for s in phn_sets:
        uid2refs.update(read_phn(f"{phn_dir}/{s}.tsv"))

    uid2hyps = {}
    for s in lab_sets:
        uid2hyps.update(
            read_lab(
                f"{lab_dir}/{s}/{lab_name}",
                pad_len,
                upsample,
            )
        )
    _main(uid2refs, uid2hyps, verbose)


def _main(uid2refs, uid2hyps, verbose):
    (p_xy, ref2pid, hyp2lid, tot, frmdiff, skipped) = comp_joint_prob(
        uid2refs, uid2hyps
    )
    ref_pur_by_hyp, ref_pur = comp_purity(p_xy, axis=0)
    hyp_pur_by_ref, hyp_pur = comp_purity(p_xy, axis=1)
    (mi, mi_norm_by_ref, mi_norm_by_hyp, h_ref, h_hyp) = comp_norm_mutual_info(p_xy)
    outputs = {
        "ref pur": ref_pur,
        "hyp pur": hyp_pur,
        "H(ref)": h_ref,
        "H(hyp)": h_hyp,
        "MI": mi,
        "MI/H(ref)": mi_norm_by_ref,
        "ref segL": comp_avg_seg_dur(uid2refs.values()),
        "hyp segL": comp_avg_seg_dur(uid2hyps.values()),
        "p_xy shape": p_xy.shape,
        "frm tot": tot,
        "frm diff": frmdiff,
        "utt tot": len(uid2refs),
        "utt miss": len(skipped),
    }
    print(tabulate([outputs.values()], outputs.keys(), floatfmt=".4f"))


if __name__ == "__main__":
    """
    compute quality of labels with respect to phone or another labels if set
    """
    parser = get_parser()
    args = parser.parse_args()

    if args.ref_lab_dir and args.ref_lab_name:
        main_lab_lab(
            args.lab_dir,
            args.lab_name,
            args.lab_sets,
            args.ref_lab_dir,
            args.ref_lab_name,
            args.pad_len,
            args.upsample,
            args.verbose,
        )
    else:
        main_phn_lab(
            args.lab_dir,
            args.lab_name,
            args.lab_sets,
            args.phn_dir,
            args.phn_sets,
            args.pad_len,
            args.upsample,
            args.verbose,
        )