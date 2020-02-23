#!/usr/bin/env python3
from espnet2.tasks.asr_hybrid import ASRHybridTask


def get_parser():
    parser = ASRHybridTask.get_parser()
    return parser


def main(cmd=None):
    r"""ASR hybrid training.

    Example:

        % python asr_hybrid_train.py asr --print_config --optim adadelta \
                > conf/train_hybrid_asr.yaml
        % python asr_hybrid_train.py --config conf/train_hybrid_asr.yaml
    """
    ASRHybridTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
