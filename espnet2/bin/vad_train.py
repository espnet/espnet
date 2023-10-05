#!/usr/bin/env python3
from espnet2.tasks.vad import VADTask


def get_parser():
    parser = VADTask.get_parser()
    return parser


def main(cmd=None):
    r"""VAD training.

    Example:

        % python vad_train.py vad --print_config --optim adadelta \
                > conf/train_vad.yaml
        % python vad_train.py --config conf/train_vad.yaml
    """
    VADTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
