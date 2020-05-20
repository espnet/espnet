#!/usr/bin/env python3
from espnet2.tasks.asr import ASRTask


def get_parser():
    parser = ASRTask.get_parser()
    return parser


def main(cmd=None):
    r"""ASR training.

    Example:

        % python asr_train.py asr --print_config --optim adadelta \
                > conf/train_asr.yaml
        % python asr_train.py --config conf/train_asr.yaml
    """
    ASRTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
