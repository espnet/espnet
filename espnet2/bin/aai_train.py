#!/usr/bin/env python3
from espnet2.tasks.aai import AAITask


def get_parser():
    parser = AAITask.get_parser()
    return parser


def main(cmd=None):
    r"""ASR training.

    Example:

        % python aai_train.py aai --print_config --optim adadelta \
                > conf/train_aai.yaml
        % python aai_train.py --config conf/train_aai.yaml
    """
    AAITask.main(cmd=cmd)


if __name__ == "__main__":
    main()
