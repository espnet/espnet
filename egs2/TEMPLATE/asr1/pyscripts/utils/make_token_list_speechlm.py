#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging
from pathlib import Path

from espnet2.speechlm.definitions import MODALITIES, special_tokens
from espnet.utils.cli_utils import get_commandline_args


def get_parser():
    parser = argparse.ArgumentParser(
        description="Build the combined vocabulary for speechlm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_json",
        type=str,
        default=[],
        nargs="+",
        help="Append json files e.g. --data_json <data_json>",
    )
    parser.add_argument(
        "--token_list_dir",
        type=Path,
        required=True,
        help="the outptu directory for the token_list and token_bias",
    )

    return parser


def main():
    logfmt = "%(asctime)s (%(module)s:%(lineno)d) %(levelname)s: %(message)s"
    logging.basicConfig(level=logging.INFO, format=logfmt)
    logging.info(get_commandline_args())

    parser = get_parser()
    args = parser.parse_args()

    # (1) find all token_list files
    all_vocab = []
    for json_file in args.data_json:
        read_handle = open(json_file)
        data_json = json.load(read_handle)
        vocabs = [vocab for vocab in data_json["vocabularies"]]
        all_vocab = all_vocab + vocabs
    logging.info(f"Find all token_list files: {all_vocab}")

    # (2) Assign each token_list file to the modality
    vocab_dict = {modality: [] for modality in MODALITIES}
    for vocab in all_vocab:
        vocab = Path(vocab)
        vocab_stem = str(vocab.stem)
        possible_match = []
        for modality in vocab_dict.keys():
            if vocab_stem.startswith(modality) and MODALITIES[modality].discrete:
                possible_match.append(modality)
        # for multiple matches, select the longest
        modality = max(possible_match, key=len)
        vocab_dict[modality].append(vocab)

    # (3) Merge the vocab file from each modality
    token_bias = {}
    token_list = []
    for modality, vocabs in vocab_dict.items():
        if len(vocabs) == 0:
            continue
        logging.info(
            f"Get vocab for modality: {modality} with token_list files: {vocabs}"
        )

        modality_vocab = {}
        for vocab in vocabs:
            try:
                this_vocab = json.load(open(vocab))
            except:
                # have to remove "\n" as text is read line-by-line
                this_vocab = [e.rstrip("\n") for e in open(vocab)]
            for e in this_vocab:
                if e in special_tokens:
                    idx = special_tokens.index(e)
                    special_tokens[idx] = e + "_<espnet>"
                    logging.warning(f"Revise special token {e} to {e}_<espnet>")
                if e not in modality_vocab:
                    modality_vocab[e] = None
                else:
                    logging.warning(f"Duplicated token: {e}. It has been seen before")

        logging.info(
            f"Modality {modality} has {len(modality_vocab)} tokens starting from {len(token_list) + len(special_tokens)}"
        )
        token_bias[modality] = len(token_list) + len(special_tokens)
        token_list = token_list + list(modality_vocab.keys())

    # (4) add special token lastly since it may be revised.
    token_list = special_tokens + token_list

    # (5) ensure each unit is unique
    seen = dict()
    for tok in token_list:
        if tok in seen:
            raise ValueError(f"token {tok} is duplicated in the token list")
        seen[tok] = None

    # (6) write vocabulary and token_bias
    vocab_writer = open(args.token_list_dir / "token_list.json", "w")
    vocab_writer.write(json.dumps(token_list, indent=4))

    bias_writer = open(args.token_list_dir / "token_bias.json", "wb")
    bias_writer.write(
        json.dumps(token_bias, indent=4, ensure_ascii=False, sort_keys=False).encode(
            "utf_8"
        )
    )


if __name__ == "__main__":
    main()
