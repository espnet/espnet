#!/usr/bin/env python3

# Copyright 2021 Jiatong Shi
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

from espnet2.tasks.diar import DiarizationTask


def get_parser():
    parser = DiarizationTask.get_parser()
    return parser


def main(cmd=None):
    r"""Speaker diarization training.

    Example:
        % python diar_train.py diar --print_config --optim adadelta \
                > conf/train_diar.yaml
        % python diar_train.py --config conf/train_diar.yaml
    """
    DiarizationTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
