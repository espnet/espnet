#!/usr/bin/env python3
from espnet2.tasks.enh_s2t import EnhS2TTask


def get_parser():
    parser = EnhS2TTask.get_parser()
    return parser


def main(cmd=None):
    r"""EnhS2T training.

    Example:

        % python enh_s2t_train.py enh_s2t --print_config --optim adadelta \
                > conf/train_enh_s2t.yaml
        % python enh_s2t_train.py --config conf/train_enh_s2t.yaml
    """
    EnhS2TTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
