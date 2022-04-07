#!/usr/bin/env python3
from espnet2.tasks.mt import MTTask


def get_parser():
    parser = MTTask.get_parser()
    return parser


def main(cmd=None):
    r"""MT training.

    Example:

        % python mt_train.py st --print_config --optim adadelta \
                > conf/train_mt.yaml
        % python mt_train.py --config conf/train_mt.yaml
    """
    MTTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
