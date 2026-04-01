#!/usr/bin/env python3
from espnet2.tasks.sot_asr import SOTASRTask


def get_parser():
    parser = SOTASRTask.get_parser()
    return parser


def main(cmd=None):
    SOTASRTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
