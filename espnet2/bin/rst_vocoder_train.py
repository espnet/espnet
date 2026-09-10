#!/usr/bin/env python3
"""Train the ESPnet-Sidon vocoder (GAN; pretrain or finetune)."""

from espnet2.tasks.rst_vocoder import RestorationVocoderTask


def get_parser():
    parser = RestorationVocoderTask.get_parser()
    return parser


def main(cmd=None):
    RestorationVocoderTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
