from argparse import ArgumentParser


def get_args():
    parser = ArgumentParser()
    parser.add_argument("text", type=str, help="text file")
    parser.add_argument(
        "--filter", type=str, help="Line filter, such as '[EN]'", default=None
    )
    return parser.parse_args()


def main():
    args = get_args()

    dur = 0.0
    with open(args.text, encoding="utf-8") as f:
        for line in f:
            if args.filter is not None and args.filter not in line:
                continue

            reco = line.split()[0]
            start, end = reco.split("-")[-1].split("_")
            start = float(start)
            end = float(end)

            dur += (end - start) / 1000.0

    print(f"{dur / 3600} hours")


if __name__ == "__main__":
    main()
