#!/usr/bin/env python3
from espnet2.tasks.gan_tts import GANTTSTask


def get_parser():
    parser = GANTTSTask.get_parser()
    return parser


def main(cmd=None):
    """GAN-based TTS training

    Example:

        % python gan_tts_train.py --print_config --optim1 adadelta
        % python gan_tts_train.py --config conf/train.yaml
    """
    GANTTSTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
