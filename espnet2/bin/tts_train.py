#!/usr/bin/env python3
from espnet2.tasks.tts import TTSTask


def get_parser():
    parser = TTSTask.add_arguments()
    return parser


def main(cmd=None):
    """

    Example:

        % python tts_train.py asr --print_config --optim adadelta
        % python tts_train.py --config conf/train_asr.yaml
    """
    TTSTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
