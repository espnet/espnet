#!/usr/bin/env python3
from espnet2.tasks.ser import SERTask


def get_parser():
    parser = SERTask.get_parser()
    return parser


def main(cmd=None):
    r"""SER training.

    Example:

        % python ser_train.py slu --print_config --optim adadelta \
                > conf/train_ser.yaml
        % python ser_train.py --config conf/train_ser.yaml
    """
    SERTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
