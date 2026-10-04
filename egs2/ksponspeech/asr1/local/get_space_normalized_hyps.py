#!/usr/bin/env python3
# encoding: utf-8 -*-

# Copyright 2020 Electronics and Telecommunications Research Institute (Jeong-Uk, Bang)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import logging
import os
import sys

import configargparse
from numpy import zeros

space_sym = "▁"
unmatched_sym = "<u>"


def get_parser():
    """Get default arguments."""
    parser = configargparse.ArgumentParser(
        description="Get space normelized hypothesis text based on the reference text.",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )

    # general configuration
    parser.add_argument(
        "--in-ref",
        type=str,
        required=True,
        help="Filename of word-level reference text",
    )
    parser.add_argument(
        "--in-hyp",
        type=str,
        required=True,
        help="Filename of word-level hypothesis text",
    )
    parser.add_argument(
        "--out-ref",
        type=str,
        required=True,
        help="Filename of space normalized reference text",
    )
    parser.add_argument(
        "--out-hyp",
        type=str,
        required=True,
        help="Filename of space normalized hypothesis text",
    )
    parser.add_argument("--verbose", "-V", type=int, default=1, help="Verbose option")
    return parser


def get_score(a, b):
    # get score for Levenshtein
    if a == b:
        return 0
    else:
        return 1


def norm_space(token):
    # get normalized token
    return token.replace(space_sym, "")


def get_norm_text(hyps, refs):
    # this implementation is modified from LevenshteinAlignment of the kaldi toolkit
    # - https://github.com/kaldi-asr/kaldi/blob/master/src/bin/align-text.cc

    # initialize variables
    hyp_norm, ref_norm = [], []

    # length of two sequences
    hlen, rlen = len(hyps), len(refs)

    # initialization
    # - this is very memory-inefficiently implemented using a vector of vectors
    scores = zeros((hlen + 1, rlen + 1))
    for r in range(0, rlen + 1):
        scores[0][r] = r
    for h in range(1, hlen + 1):
        scores[h][0] = scores[h - 1][0] + 1
        for r in range(1, rlen + 1):
            hyp_nosp, ref_nosp = norm_space(hyps[h - 1]), norm_space(refs[r - 1])
            sub_or_cor = scores[h - 1][r - 1] + get_score(hyp_nosp, ref_nosp)
            insert, delete = scores[h - 1][r] + 1, scores[h][r - 1] + 1
            scores[h][r] = min(sub_or_cor, insert, delete)

    # traceback and compute the alignment
    h, r = hlen, rlen  # start from the bottom
    while h > 0 or r > 0:
        if h == 0:
            last_h, last_r = h, r - 1
        elif r == 0:
            last_h, last_r = h - 1, r
        else:
            # get score
            hyp_nosp, ref_nosp = norm_space(hyps[h - 1]), norm_space(refs[r - 1])
            sub_or_cor = scores[h - 1][r - 1] + get_score(hyp_nosp, ref_nosp)
            insert, delete = scores[h - 1][r] + 1, scores[h][r - 1] + 1

            # choose sub_or_cor if all else equal
            if sub_or_cor <= min(insert, delete):
                last_h = h - 1
                last_r = r - 1
            else:
                if insert < delete:
                    last_h = h - 1
                    last_r = r
                else:
                    last_h = h
                    last_r = r - 1

        c_hyp = hyps[last_h] if last_h != h else ""
        c_ref = refs[last_r] if last_r != r else ""
        h, r = last_h, last_r

        # do word-spacing normalization
        if c_hyp != c_ref and norm_space(c_hyp) == norm_space(c_ref):
            c_hyp = c_ref
        if c_hyp != "":
            hyp_norm.append(c_hyp)
        if c_ref != "":
            ref_norm.append(c_ref)

    # reverse list
    hyp_norm.reverse()
    ref_norm.reverse()

    return (hyp_norm, ref_norm)


def main(args):
    """Run the main normalizing function."""
    parser = get_parser()
    args = parser.parse_args(args)

    # logging info
    if args.verbose == 1:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    elif args.verbose == 2:
        logging.basicConfig(
            level=logging.DEBUG,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
    else:
        logging.basicConfig(
            level=logging.WARN,
            format="%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s",
        )
        logging.warning("Skip DEBUG/INFO messages")

    # check input arguments
    if not os.path.exists(args.in_ref):
        logging.error("Filename '" + args.in_ref + "' is not exist")
        logging.error("Please check the '--in-ref' argument")
        sys.exit(1)
    else:
        logging.info("Loading reference text: " + args.in_ref)
    if not os.path.exists(args.in_hyp):
        logging.error("Filename '" + args.in_hyp + "' is not exist")
        logging.error("Please check the '--raw-trans' argument")
        sys.exit(1)
    else:
        logging.info("Loading hypothesis text: " + args.in_hyp)

    # read lines of ref & hyp files
    refs = open(args.in_ref, "r").readlines()
    hyps = open(args.in_hyp, "r").readlines()

    # comparing the number of files
    if len(refs) != len(hyps):
        logging.error(
            "# of sentences in refs(%s) and hyps(%s) are different."
            % (str(len(refs)), str(len(hyps)))
        )
        sys.exit(1)

    # create normalized ref & hyp files
    save_norm_ref = open(args.out_ref, mode="w")
    save_norm_hyp = open(args.out_hyp, mode="w")

    # run Levenshtein alignment processing
    for i, _ in enumerate(refs):
        # get utt-id
        ref_pos = refs[i].rindex("(")
        hyp_pos = hyps[i].rindex("(")
        ref_utt_id = refs[i][ref_pos + 1 :].strip()[:-1]
        hyp_utt_id = hyps[i][hyp_pos + 1 :].strip()[:-1]

        # check the number of line in two texts
        if ref_utt_id != hyp_utt_id:
            logging.error(
                "The utt-id is not sorted. (ref[%s:%s], hyp[%s:%s])"
                % (str(ref_utt_id), str(i), str(hyp_utt_id), str(i))
            )
            sys.exit(1)

        # do character-level tokenizing with space symbol ('_')
        ref = (
            " ".join(space_sym + refs[i][:ref_pos].strip().replace(" ", space_sym))
            .replace(space_sym + " ", space_sym)
            .split()
        )
        hyp = (
            " ".join(space_sym + hyps[i][:hyp_pos].strip().replace(" ", space_sym))
            .replace(space_sym + " ", space_sym)
            .split()
        )

        # get space-normalized texts with Levenshtein algorithm
        hyp_norm, ref_norm = get_norm_text(hyp, ref)

        # save space-normalized texts
        save_norm_hyp.write(str(" ".join(hyp_norm)).strip() + "(" + hyp_utt_id + ")\n")
        save_norm_ref.write(str(" ".join(ref_norm)).strip() + "(" + ref_utt_id + ")\n")

    # close files
    save_norm_ref.close()
    save_norm_hyp.close()

    logging.info("Succeeded creating normalized texts.")


if __name__ == "__main__":
    main(sys.argv[1:])
