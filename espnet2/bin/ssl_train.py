#!/usr/bin/env python3
from espnet2.tasks.ssl import SSLTask


def get_parser():
    parser = SSLTask.get_parser()
    return parser


def main(cmd=None):
    """Hubert pretraining.

    Example:
        % python ssl_train.py asr --print_config --optim adadelta \
                > conf/hubert_asr.yaml
        % python ssl_train.py --config conf/train_asr.yaml
    """
    SSLTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
