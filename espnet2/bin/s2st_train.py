#!/usr/bin/env python3
from espnet2.tasks.s2st import S2STTask


def get_parser():
    parser = S2STTask.get_parser()
    return parser


def main(cmd=None):
    r"""S2ST training.

    Example:

        % python s2st_train.py s2st --print_config --optim adadelta \
                > conf/train_s2st.yaml
        % python s2st_train.py --config conf/train_s2st.yaml
    """
    S2STTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
