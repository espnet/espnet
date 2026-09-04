#!/usr/bin/env python3
"""Train the ESPnet-Sidon w2v-BERT 2.0 feature predictor."""

from espnet2.tasks.sidon import SidonTask


def get_parser():
    parser = SidonTask.get_parser()
    return parser


def main(cmd=None):
    SidonTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
