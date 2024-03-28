#!/usr/bin/env python3
from espnet2.tasks.tts2 import TTS2Task


def get_parser():
    parser = TTS2Task.get_parser()
    return parser


def main(cmd=None):
    """TTS training

    Example:

        % python tts2_train.py asr --print_config --optim adadelta
        % python tts2_train.py --config conf/train_tts2.yaml
    """
    TTS2Task.main(cmd=cmd)


if __name__ == "__main__":
    main()
