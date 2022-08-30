import argparse


def get_parser():
    parser = argparse.ArgumentParser(
        description="Calculate the diagonality of the self-attention weights."
    )
    parser.add_argument(
        "--src",
        type=str,
        help="path to results"
    )
    return parser

def read_src(src):
    lines = open(src, "r").readlines()
    data = [x.split() for x in lines]
    return [float(x[-1]) for x in data]

if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    data = read_src(args.src)
    res = sum(data) / len(data)

    print(args.src, "avg RTF", str(res))