#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import argparse
import json
import logging
from pathlib import Path

from espnet2.speechlm.definitions import modalities, special_tokens


def get_parser():
    parser = argparse.ArgumentParser(
        description="Build the combined vocabulary for speechlm",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_json",
        type=str,
        default=[],
        action="append",
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
    parser = get_parser()
    args = parser.parse_args()

    # (1) find all token_list files
    all_vocab = []
    for json_file in args.data_json:
        read_handle = open(json_file)
        all_vocab = all_vocab + json.load(read_handle)["vocabularies"]
    logging.info(f"Find all token_list files: {all_vocab}")

    # (2) Assign each token_list file to the modality
    vocab_dict = {modality: [] for modality in modalities}
    for vocab in all_vocab:
        vocab = Path(vocab)
        vocab_stem = str(vocab.stem)
        for modality in vocab_dict.keys():
            if vocab_stem.startswith(modality) and modalities[modality].discrete:
                vocab_dict[modality].append(vocab)

    # (3) First include special tokens.
    token_list = special_tokens.copy()

    # (4) Merge the vocab file from each modality
    token_bias = {}
    for modality, vocabs in vocab_dict.items():
        if len(vocabs) == 0:
            continue
        logging.info(
            f"Get vocab for modality: {modality} with token_list files: {vocabs}"
        )

        modality_vocab = []
        for vocab in vocabs:
            this_vocab = [e.rstrip("\n") for e in open(vocab)]
            for e in this_vocab:
                if e not in modality_vocab:
                    modality_vocab.append(e)
                else:
                    logging.warning(f"Duplicated token: {e}. It has been seen before")

        logging.info(
            f"Modality has {len(modality_vocab)} starting from {len(token_list)}"
        )
        token_bias[modality] = len(token_list)
        token_list = token_list + modality_vocab

    vocab_writer = open(args.token_list_dir / "token_list", "w")
    for tok in token_list:
        vocab_writer.write(f"{tok}\n")

    # (5) write vocabulary and token_bias
    bias_writer = open(args.token_list_dir / "token_bias.json", "wb")
    bias_writer.write(
        json.dumps(token_bias, indent=4, ensure_ascii=False, sort_keys=False).encode(
            "utf_8"
        )
    )


if __name__ == "__main__":
    main()
