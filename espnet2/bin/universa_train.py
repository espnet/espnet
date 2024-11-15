#!/usr/bin/env python3
from espnet2.tasks.universa import UniversaTask

def get_parser():
    parser = UniversaTask.get_parser()
    return parser

def main(cmd=None):
    """Universa training

    Example:

        % python universa_train.py universa --print_config --optim adadelta
        % python universa_train.py --config conf/train_universa.yaml
    """
    UniversaTask.main(cmd=cmd)

if __name__ == "__main__":
    main()