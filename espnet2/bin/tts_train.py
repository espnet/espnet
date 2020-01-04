#!/usr/bin/env python3
from espnet2.tasks.tts import TTSTask


def get_parser():
    parser = TTSTask.get_parser()
    return parser


def main(cmd=None):
    """TTS training

    Example:

        % python tts_train.py asr --print_config --optim adadelta
        % python tts_train.py --config conf/train_asr.yaml
    """
    TTSTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
