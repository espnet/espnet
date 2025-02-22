#!/usr/bin/env python3
from espnet2.tasks.speechlm import SpeechLMTask


def get_parser():
    parser = SpeechLMTask.get_parser()
    return parser


def main(cmd=None):
    """Speech LM training.

    Example:

        % python speechlm_train.py --print_config --optim adadelta
        % python speechlm_train.py --config conf/train.yaml
    """
    SpeechLMTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
