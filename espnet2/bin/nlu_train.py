#!/usr/bin/env python3
from espnet2.tasks.nlu import NLUTask


def get_parser():
    parser = NLUTask.get_parser()
    return parser


def main(cmd=None):
    r"""NLU training.

    Example:

        % python nlu_train.py st --print_config --optim adadelta \
                > conf/train_nlu.yaml
        % python nlu_train.py --config conf/train_nlu.yaml
    """
    NLUTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
