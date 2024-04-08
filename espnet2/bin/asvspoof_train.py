#!/usr/bin/env python3
from espnet2.tasks.asvspoof import ASVSpoofTask


def get_parser():
    parser = ASVSpoofTask.get_parser()
    return parser


def main(cmd=None):
    r"""ASVSpoof training.

    Example:
        % python asvspoof_train.py asr --print_config --optim adadelta \
                > conf/train_asvspoof.yaml
        % python asvspoof_train.py --config conf/train_asvspoof.yaml
    """
    ASVSpoofTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
