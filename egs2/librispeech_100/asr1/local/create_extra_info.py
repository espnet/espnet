import argparse
import os
from typing import List

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
        "--token_lists",
        type=str,
        metavar="N",
        nargs="*",
        help="The token list for the tokenizer",
    )
    parser.add_argument(
        "--merged_token_list",
        type=str,
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
        "--token_types",
        type=str,
        metavar="N",
        default=["bpe"],
        nargs="+",
        help="The token types of the external text data",
    )

    return parser.parse_args()


def get_vocab_size(token_list: str) -> int:
    vocab_size = 0
    with open(token_list, "r", encoding="utf-8") as token_list_file:
        lines = token_list_file.readlines()
        vocab_size = len(lines)

    return vocab_size


def text2char(text: List[str]) -> List[str]:
    text = " ".join(text)

    chars = []
    for char in text:
        if char == " ":
            chars.append("<space>")
        else:
            chars.append(char)

    return chars


if __name__ == "__main__":
    args = get_args()

    if args.token_types == ["speech"]:
        pseudo_labels = {}
        with open(
            "/ocean/projects/cis210027p/jiatong/discreate_asr/pseudo_label/wavlm_large/train_960/pseudo_labels_km1000.txt",
            "r",
            encoding="utf-8",
        ) as pseudo_labels_file:
            pseudo_labels_lines = pseudo_labels_file.readlines()
            for pseudo_labels_line in pseudo_labels_lines:
                pseudo_labels_line = pseudo_labels_line.strip()
                pseudo_labels_line = pseudo_labels_line.split()

                uid = "speech_injection-" + pseudo_labels_line[0]
                tokens = pseudo_labels_line[1:]

                pseudo_labels[uid] = tokens

        # hard-coded
        for file in args.files:
            with open(
                f"{args.asr_data_dir}/{file}",
                "r",
                encoding="utf-8",
            ) as extra_file, open(
                f"{args.asr_stats_dir}/train/speech_injection_shape",
                "w+",
                encoding="utf-8",
            ) as token_stats_file:
                lines = extra_file.readlines()
                for line in tqdm(lines):
                    line = line.strip()
                    line = line.split()

                    uid = line[0]
                    text = pseudo_labels[uid]

                    text_len = len(text)
                    token_stats_file.write(f"{uid} {text_len}\n")
    else:
        # merge token_list
        all_token_list = set()
        for token_list in args.token_lists:
            with open(token_list, "r", encoding="utf-8") as token_list_file:
                lines = token_list_file.readlines()
                for line in lines:
                    line = line.strip()
                    if line == "":
                        continue

                    all_token_list.update([line])

        all_token_list.remove("<sos/eos>")
        all_token_list = list(all_token_list) + ["<sos/eos>"]

        merged_token_list_folder = "/".join(args.merged_token_list.split("/")[:-1])
        if not os.path.exists(merged_token_list_folder):
            os.makedirs(merged_token_list_folder)

        with open(
            args.merged_token_list,
            "w+",
            encoding="utf-8",
        ) as merged_token_list_file:
            for token in all_token_list:
                token = token.strip()
                merged_token_list_file.write(f"{token}\n")

        for file in args.files:
            with open(
                f"{args.asr_data_dir}/{file}",
                "r",
                encoding="utf-8",
            ) as extra_file, open(
                f"{args.asr_stats_dir}/train/{file}_shape",
                "w+",
                encoding="utf-8",
            ) as token_stats_file:
                extra_file_lines = extra_file.readlines()
                for extra_file_line in tqdm(extra_file_lines):
                    extra_file_line = extra_file_line.strip()
                    extra_file_line = extra_file_line.split()

                    uid = extra_file_line[0]
                    text = extra_file_line[1:]

                    char = text2char(text)

                    # need to use char as it is the longest sequence
                    text_len = len(char)

                    token_stats_file.write(f"{uid} {text_len}\n")
