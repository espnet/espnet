from pathlib import Path

import numpy as np

from espnet2.text.build_tokenizer import build_tokenizer
from espnet2.text.token_id_converter import TokenIDConverter


def parse_owsm(
    text_in,
    text_out,
    tokenizer,
    token_id_converter,
):
    first_time = token_id_converter.token2id["<0.00>"]
    last_time = token_id_converter.token2id["<30.00>"]

    with open(text_in, "r") as fin, open(text_out, "w") as fout:
        for line in fin:
            utt_id, text = line.strip().split(maxsplit=1)

            tokens = tokenizer.text2tokens(text)
            text_ints = np.array(token_id_converter.tokens2ids(tokens))

            text_ints = text_ints[
                np.logical_or(
                    text_ints < first_time,
                    text_ints > last_time,
                )
            ]

            tokens = token_id_converter.ids2tokens(text_ints)
            text = tokenizer.tokens2text(tokens)

            fout.write(f"{utt_id} {text}\n")


if __name__ == "__main__":
    root = "dump/raw"

    tokenizer = build_tokenizer(
        token_type="bpe",
        bpemodel="owsm_v3.1/s2t1/data/token_list/bpe_unigram50000/bpe.model",
    )

    token_id_converter = TokenIDConverter(
        token_list="owsm_v3.1/s2t1/data/token_list/bpe_unigram50000/tokens.txt",
        unk_symbol="<unk>",
    )

    for name in ["train_v3", "dev_v3"]:
        (Path(root) / name / "text").rename(Path(root) / name / "text.old")
        parse_owsm(
            Path(root) / name / "text.old",
            Path(root) / name / "text",
            tokenizer,
            token_id_converter,
        )
