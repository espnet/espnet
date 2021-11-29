#!/usr/bin/env python3
from espnet2.tasks.vc import VCTask


def get_parser():
    parser = VCTask.get_parser()
    return parser


def main(cmd=None):
    """VC training

    Example:

        % python vc_train.py asr --print_config --optim adadelta
        % python vc_train.py --config conf/train_asr.yaml
    """
    VCTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
