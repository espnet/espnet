#!/usr/bin/env python3
"""Train the ESPnet-Sidon flow-matching vocoder."""

from espnet2.tasks.sidon_flow_vocoder import SidonFlowVocoderTask


def get_parser():
    parser = SidonFlowVocoderTask.get_parser()
    return parser


def main(cmd=None):
    SidonFlowVocoderTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
