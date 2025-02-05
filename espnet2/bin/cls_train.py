#!/usr/bin/env python3
from espnet2.tasks.cls import CLSTask


def get_parser():
    parser = CLSTask.get_parser()
    return parser


def main(cmd=None):
    r"""Classification training.

    Example:
        % python cls_train.py asr --print_config --optim adadelta \
                > conf/train_cls.yaml
        % python cls_train.py --config conf/train_cls.yaml
    """
    CLSTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
