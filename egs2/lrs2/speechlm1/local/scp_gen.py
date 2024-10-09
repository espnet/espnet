import argparse
import glob
import os


def main():
    parser = get_parser()
    args = parser.parse_args()
    pretrain_file_lists = sorted(
        glob.glob(
            os.path.join(args.data_dir, "pretrain", "**", "*.mp4"), recursive=True
        )
    )
    pretrain_file_lists = [
        f"{os.sep}".join(os.path.normpath(f).split(os.sep)[-3:])[:-4]
        for f in pretrain_file_lists
    ]
    train_file_lists = sorted(
        glob.glob(
            os.path.join(args.data_dir, "trainval", "**", "*.mp4"), recursive=True
        )
    )
    train_file_lists = [
        f"{os.sep}".join(os.path.normpath(f).split(os.sep)[-3:])[:-4]
        for f in train_file_lists
    ]
    test_file_lists = sorted(
        glob.glob(os.path.join(args.data_dir, "test", "**", "*.mp4"), recursive=True)
    )
    test_file_lists = [
        f"{os.sep}".join(os.path.normpath(f).split(os.sep)[-3:])[:-4]
        for f in test_file_lists
    ]
    with open("local/lrs3-valid.id", "r") as txt:
        lines = txt.readlines()
    val_file_lists = [line.strip() for line in lines]
    train_file_lists = [f for f in train_file_lists if f not in val_file_lists]

    train_out_dir = f"data/train/video.scp"
    os.makedirs(os.path.dirname(train_out_dir), exist_ok=True)
    with open(train_out_dir, "w") as txt:
        for line in train_file_lists + pretrain_file_lists:
            txt.write(
                f'{"_".join(line.split(os.sep))} \
                    {os.path.join(args.data_dir, line)}.mp4\n'
            )

    val_out_dir = f"data/val/video.scp"
    os.makedirs(os.path.dirname(val_out_dir), exist_ok=True)
    with open(val_out_dir, "w") as txt:
        for line in val_file_lists:
            txt.write(
                f'{"_".join(line.split(os.sep))} \
                    {os.path.join(args.data_dir, line)}.mp4\n'
            )

    test_out_dir = f"data/test/video.scp"
    os.makedirs(os.path.dirname(test_out_dir), exist_ok=True)
    with open(test_out_dir, "w") as txt:
        for line in test_file_lists:
            txt.write(
                f'{"_".join(line.split(os.sep))} \
                    {os.path.join(args.data_dir, line)}.mp4\n'
            )

    with open(f"data/train/text", "w") as txtw:
        for line in train_file_lists + pretrain_file_lists:
            with open(os.path.join(args.data_dir, line) + ".txt", "r") as txt:
                txtw.write(
                    f'{"_".join(line.split(os.sep))}\
                        \t{txt.readlines()[0][7:].strip()}\n'
                )

    with open(f"data/val/text", "w") as txtw:
        for line in val_file_lists:
            with open(os.path.join(args.data_dir, line) + ".txt", "r") as txt:
                txtw.write(
                    f'{"_".join(line.split(os.sep))}\
                        \t{txt.readlines()[0][7:].strip()}\n'
                )

    with open(f"data/test/text", "w") as txtw:
        for line in test_file_lists:
            with open(os.path.join(args.data_dir, line) + ".txt", "r") as txt:
                txtw.write(
                    f'{"_".join(line.split(os.sep))}\
                        \t{txt.readlines()[0][7:].strip()}\n'
                )


def get_parser():
    parser = argparse.ArgumentParser(
        description="Command-line script for preprocessing."
    )
    parser.add_argument("--data_dir", type=str, required=True, help="data_dir")
    parser.add_argument(
        "--model", type=str, required=True, help="AV-HuBERT model config"
    )
    return parser


if __name__ == "__main__":
    main()
