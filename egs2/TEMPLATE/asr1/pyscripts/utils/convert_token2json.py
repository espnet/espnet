import argparse
import json
import os
from io import BytesIO

import kaldiio
import soundfile as sf

if __name__ == "__main__":
    parser = argparse.ArgumentParser("convert espnet tokens for bitrate calculation")
    parser.add_argument("--vocab", type=str, required=True)
    parser.add_argument("--token", type=str, required=True)
    parser.add_argument("--ref_scp", type=str, required=True)
    parser.add_argument("--result_dir", type=str, required=True)

    args = parser.parse_args()

    if not os.path.exists(args.result_dir):
        os.makedirs(args.result_dir)

    vocab_json = {}
    token_json = {}
    ref_len_file = open(
        os.path.join(args.result_dir, "ref_len.scp"), "w", encoding="utf-8"
    )

    with open(args.vocab, "r", encoding="utf-8") as vocab_f, open(
        args.token, "r", encoding="utf-8"
    ) as token_f, open(args.ref_scp, "r", encoding="utf-8") as ref_f:

        # preparing vocabulary
        vocab_json[0] = vocab_f.read().strip().split("\n")
        json.dump(
            vocab_json,
            open(os.path.join(args.result_dir, "vocab.json"), "w", encoding="utf-8"),
        )

        # preparing reference length
        for line in ref_f:
            if len(line.strip()) == 0:
                break
            key, value = line.strip().split(maxsplit=1)
            if value.endswith("|"):
                # Streaming input e.g. cat a.wav |
                with kaldiio.open_like_kaldi(value, "rb") as wav_f:
                    with BytesIO(wav_f.read()) as g:
                        data, sample_rate = sf.read(g)
            else:
                data, sample_rate = sf.read(value)
            ref_len_file.write("{} {}\n".format(key, len(data) / sample_rate))

        # preparing tokens
        for line in token_f:
            if len(line.strip()) == 0:
                break
            key, value = line.strip().split(maxsplit=1)
            token_json[key] = [
                value.split(),
            ]
        json.dump(
            token_json,
            open(os.path.join(args.result_dir, "tokens.json"), "w", encoding="utf-8"),
        )
