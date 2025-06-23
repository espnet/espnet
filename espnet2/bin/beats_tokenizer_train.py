#!/usr/bin/env python3
from espnet2.tasks.beats import BeatsTokenizerTask


def get_parser():
    parser = BeatsTokenizerTask.get_parser()
    return parser


def main(cmd=None):
    """Beats Tokenizer pretraining.

    Example:
        % python beats_tokenizer_train.py asr --print_config \
                --optim adadelta > conf/beats_tokenizer_train.yaml
        % python beats_tokenizer_train.py --config conf/beats_tokenizer_train.yaml
    """
    BeatsTokenizerTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
