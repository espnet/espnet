import argparse
import os


def gen_new_segments(datadir, spk_list):
    if not os.path.isfile(os.path.join(datadir, "segments")):
        raise ValueError("no segments file found in datadir")

    new_segments = open(os.path.join(datadir, "new_segments"), "w", encoding="utf-8")
    segments = open(os.path.join(datadir, "segments"), "r", encoding="utf-8")
    while True:
        line = segments.readline()
        if not line:
            break
        spk = line.split("_")[0]
        if spk in spk_list:
            new_segments.write(line)
    new_segments.close(), segments.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", "-s", type=str, help="source data directory")
    parser.add_argument("--conf", "-c", type=str, help="split by speaker")
    parser.add_argument("--tag", "-t", type=str, help="the prefix of data spit result")
    parser.add_argument(
        "--train", type=str, default="", help="specific name for training dir"
    )
    parser.add_argument("--dev", type=str, default="", help="specific name for dev dir")
    parser.add_argument(
        "--test", type=str, default="", help="specific name for test dir"
    )
    args = parser.parse_args()

    with open(args.conf, "r", encoding="utf-8") as f:
        f_content = f.read().strip().split("\n")
        split_info = {}
        for line in f_content:
            line = line.split(",")
            split_info[line[0]] = line[1:]

    # construct dataset
    train_dir = (
        "data/train_{}".format(args.source)
        if args.train == ""
        else "data/{}".format(args.train)
    )
    test_dir = (
        "data/test_{}".format(args.source)
        if args.test == ""
        else "data/{}".format(args.test)
    )
    dev_dir = (
        "data/dev_{}".format(args.source)
        if args.dev == ""
        else "data/{}".format(args.dev)
    )

    gen_new_segments(train_dir, split_info["train"])
    gen_new_segments(test_dir, split_info["test"])
    gen_new_segments(dev_dir, split_info["dev"])
