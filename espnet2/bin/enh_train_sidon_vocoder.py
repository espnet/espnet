#!/usr/bin/env python3
"""Train the ESPnet-Sidon vocoder (GAN; pretrain or finetune)."""

from espnet2.tasks.sidon_vocoder import SidonVocoderTask


def get_parser():
    parser = SidonVocoderTask.get_parser()
    return parser


def main(cmd=None):
    SidonVocoderTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
