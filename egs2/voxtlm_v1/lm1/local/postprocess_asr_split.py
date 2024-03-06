from argparse import ArgumentParser
from pathlib import Path
from typing import Optional


def process_text(fin: Path, fout: Path, sos: str, prefix: str):
    # Convert text (hyp or ref) to transcript format

    hyp_comb = {}
    with fin.open("r") as fp_in, fout.open("w") as fp_out:
        for line in fp_in.readlines():
            if prefix is None or line.startswith(prefix):
                uid = line.split(maxsplit=1)[0]

                # check here:
                # if not split - do below
                uid_split = uid.split("_")
                trans = line.split(sos)[-1].strip()

                # print(uid, len(uid_split), uid_split)
                if len(uid_split) > 2:
                    # audio is split

                    uid = "_".join(uid_split[:-1])
                    count = int(uid_split[2])

                    if uid not in hyp_comb:
                        hyp_comb[uid] = [trans]
                    else:
                        hyp_comb[uid].append(trans)

                    # print(uid, count, hyp_comb[uid])
                else:
                    # trans = line.split(sos)[-1].strip()
                    fp_out.write(f"{trans}\t({uid})\n")

        # write comb ones
        for k in hyp_comb.keys():

            trans = " ".join(hyp_comb[k])
            print("Combine:", k, " ", trans)
            fp_out.write(f"{trans}\t({k})\n")


def get_parser():
    parser = ArgumentParser(description="Convert text to transcript format")

    parser.add_argument("-i", "--input", type=Path, required=True, help="Input file")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output file")
    parser.add_argument(
        "-s",
        "--sos",
        type=str,
        required=True,
        help="The symbol to split each line of text.",
    )
    parser.add_argument(
        "-p",
        "--prefix",
        type=str,
        required=True,
        help="Only keep lines with the specified prefix.",
    )

    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    process_text(args.input, args.output, args.sos, args.prefix)
