from espnet2.tasks.lid import LIDTask


def get_parser():
    parser = LIDTask.get_parser()
    return parser


def main(cmd=None):
    r"""LID training."""
    LIDTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
