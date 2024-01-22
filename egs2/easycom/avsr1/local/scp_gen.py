import argparse
import glob
import os


def main():
    parser = get_parser()
    args = parser.parse_args()
    splits = ["train", "val", "test"]
    args.LRS3 = True if args.LRS3 == "True" else False
    args.include_wearer = True if args.include_wearer == "True" else False
    for split in splits:
        file_lists = sorted(
            glob.glob(
                os.path.join(args.data_dir, split, "Audio", "*.wav"),
            )
        )

        if not args.include_wearer and split != "train":
            file_lists = [file for file in file_lists if int(file.split("_")[-3]) != 2]

        file_lists = [
            f"{os.sep}".join(os.path.normpath(file).split(os.sep)[-3:])[:-4]
            for file in file_lists
        ]

        audio_scp = (
            f"data/{split}_with_LRS3/wav.scp" if args.LRS3 else f"data/{split}/wav.scp"
        )
        os.makedirs(os.path.dirname(audio_scp), exist_ok=True)
        with open(audio_scp, "w") as txt:
            for line in file_lists:
                txt.write(
                    f'{"_".join(line.split(os.sep))} \
                        {os.path.join(args.data_dir, line)}.wav\n'
                )

        text_scp = f"data/{split}_with_LRS3/text" if args.LRS3 else f"data/{split}/text"
        with open(text_scp, "w") as txtw:
            for line in file_lists:
                with open(
                    os.path.join(args.data_dir, line.replace("Audio", "Text")) + ".txt",
                    "r",
                ) as txt:
                    txtw.write(
                        f'{"_".join(line.split(os.sep))}\
                            \t{txt.readlines()[0].strip()}\n'
                    )

        if args.LRS3 and split == "train":
            file_lists = sorted(
                glob.glob(
                    os.path.join(args.data_dir, "LRS3", "Audio", split, "*", "*.wav")
                )
            )
            file_lists = [
                f"{os.sep}".join(os.path.normpath(file).split(os.sep)[-3:])[:-4]
                for file in file_lists
            ]
            with open(audio_scp, "a") as txt:
                for line in file_lists:
                    txt.write(
                        f'{"_".join(line.split(os.sep))} \
                            {os.path.join(args.data_dir, "LRS3", "Audio", line)}.wav\n'
                    )

            with open(text_scp, "a") as txtw:
                for line in file_lists:
                    with open(
                        os.path.join(args.data_dir, "LRS3", "Text", line + ".txt"), "r"
                    ) as txt:
                        txtw.write(
                            f'{"_".join(line.split(os.sep))}\
                                \t{txt.readlines()[0].strip()}\n'
                        )


def get_parser():
    parser = argparse.ArgumentParser(
        description="Command-line script for scp generation."
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="preprocessed_data_dir"
    )
    parser.add_argument(
        "--include_wearer",
        type=str,
        required=True,
        help="wheter include the eyeglasses wearer",
    )
    parser.add_argument(
        "--LRS3", type=str, default=False, help="whether use LRS3 for training"
    )
    return parser


if __name__ == "__main__":
    main()
