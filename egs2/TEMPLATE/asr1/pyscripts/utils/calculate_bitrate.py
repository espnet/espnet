# Official Bitrate Calculation for
# Interspeech2024 Discrete Speech Challenge

import argparse
import json
import math

import numpy as np

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Bitrate Calculation (at seconds)")
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--tokens", type=str, required=True)
    parser.add_argument("--reference_len", type=str, required=True)
    parser.add_argument("--bitrate_details", type=str, default=None)
    parser.add_argument("--bitrate_result", type=str, default=None)

    args = parser.parse_args()

    reference_len_dict = {}

    with open(args.reference_len, "r", encoding="utf-8") as ref_f:
        for line in ref_f:
            if len(line.strip()) == 0:
                break
            key, value = line.strip().split(maxsplit=1)
            reference_len_dict[key] = float(value)

    with open(args.vocab, "r", encoding="utf-8") as vocab_f, open(
        args.tokens, "r", encoding="utf-8"
    ) as tokens_f:

        if args.bitrate_details is not None:
            bitrate_details_f = open(args.bitrate_details, "w", encoding="utf-8")

        vocab = json.load(vocab_f)
        tokens = json.load(tokens_f)

        for key in vocab.keys():
            vocab[key] = math.log2(len(vocab[key]))  # convert to log space

        bitrates = []

        for key in tokens.keys():
            assert (
                key in reference_len_dict.keys()
            ), "mismatched key ({}) between reference and provided tokens".format(key)
            ref_len = reference_len_dict[key]  # in seconds
            token = tokens[key]
            cum_info = 0
            for stream in range(len(token)):
                assert (
                    str(stream) in vocab.keys()
                ), "stream {} does not have vocab, check vocab file {}".format(
                    stream,
                    args.vocab,
                )
                info = len(token[stream]) / ref_len * vocab[str(stream)]
                cum_info += info
            bitrates.append(cum_info)
            if args.bitrate_details is not None:
                bitrate_details_f.write("{} {}\n".format(key, cum_info))

    final_bitrates = np.mean(bitrates)
    if args.bitrate_result is not None:
        bitrate_result_f = open(args.bitrate_result, "w", encoding="utf-8")
        bitrate_results_f.write("bitrate: {}".format(final_bitrates))

    print("final bitrates for the data: {}".format(final_bitrates))
