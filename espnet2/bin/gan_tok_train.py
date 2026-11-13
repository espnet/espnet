#!/usr/bin/env python3
from espnet2.tasks.gan_tok import GANSpeechTokenizerTask


def get_parser():
    parser = GANSpeechTokenizerTask.get_parser()
    return parser


def main(cmd=None):
    """GAN-based speech tokenizer training"""
    GANSpeechTokenizerTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
