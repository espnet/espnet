#!/usr/bin/env python3
from espnet2.tasks.gan_codec import GANCodecTask


def get_parser():
    parser = GANCodecTask.get_parser()
    return parser


def main(cmd=None):
    """GAN-based Codec training

    Example:

        % python gan_codec_train.py --print_config --optim1 adadelta
        % python gan_codec_train.py --config conf/train.yaml
    """
    GANCodecTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
