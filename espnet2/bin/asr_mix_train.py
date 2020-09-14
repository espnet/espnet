#!/usr/bin/env python3
from espnet2.tasks.asr_mix import ASRMixTask


def get_parser():
    parser = ASRMixTask.get_parser()
    return parser


def main(cmd=None):
    r"""ASR training.

    Example:

        % python asr_train.py asr --print_config --optim adadelta \
                > conf/train_asr.yaml
        % python asr_train.py --config conf/train_asr.yaml
    """
    ASRMixTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
