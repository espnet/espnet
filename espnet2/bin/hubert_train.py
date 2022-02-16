#!/usr/bin/env python3
from espnet2.tasks.hubert import HubertTask


def get_parser():
    parser = HubertTask.get_parser()
    return parser


def main(cmd=None):
    """Hubert pretraining.

    Example:
        % python hubert_train.py asr --print_config --optim adadelta \
                > conf/hubert_asr.yaml
        % python hubert_train.py --config conf/train_asr.yaml
    """
    HubertTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
