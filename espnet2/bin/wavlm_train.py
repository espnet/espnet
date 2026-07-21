#!/usr/bin/env python3
from espnet2.tasks.wavlm import WavLMTask


def get_parser():
    parser = WavLMTask.get_parser()
    return parser


def main(cmd=None):
    """WavLM pretraining.

    Example:
        % python wavlm_train.py asr --print_config --optim adadelta \
                > conf/hubert_asr.yaml
        % python wavlm_train.py --config conf/train_asr.yaml
    """
    WavLMTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
