import argparse
from tqdm import tqdm


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract shape info for extra files")

    parser.add_argument(
        "-f",
        "--files",
        nargs="+",
        help="Extra files that needs to be processed",
        required=True,
    )
    parser.add_argument(
        "--asr_data_dir",
        type=str,
        help="The path for data path",
    )
    parser.add_argument(
        "--asr_stats_dir",
        type=str,
        help="The path for asr stats",
    )
    parser.add_argument(
        "--token_type",
        type=str,
        default="bpe",
        choices=[
            "bpe",
            "phn",
        ],
        help="The text will be tokenized " "in the specified level token",
    )
    parser.add_argument(
        "--bpe_model",
        type=str,
        default=None,
        help="The model file of sentencepiece",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    if args.token_type == "bpe":
        import sentencepiece as spm
        sp = spm.SentencePieceProcessor(args.bpe_model)
        vocab_size = sp.get_piece_size()
        for file in args.files:
            with open(
                f"{args.asr_data_dir}/{file}",
                "r", encoding="utf-8",
            ) as extra_file, open(
                f"{args.asr_stats_dir}/train/{file}_shape.bpe",
                "w+", encoding="utf-8",
            ) as bpe_stats_file:
                extra_file_lines = extra_file.readlines()

                for extra_file_line in tqdm(extra_file_lines):
                    extra_file_line = extra_file_line.strip()
                    extra_file_line = extra_file_line.split(" ")

                    uid = extra_file_line[0]
                    text = " ".join(extra_file_line[1:])

                    len_bpe = len(sp.encode(text))

                    bpe_stats_file.write(f"{uid} {len_bpe},{vocab_size}\n")
    else:
        pass
