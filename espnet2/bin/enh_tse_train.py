#!/usr/bin/env python3
from espnet2.tasks.enh_tse import TargetSpeakerExtractionTask


def get_parser():
    parser = TargetSpeakerExtractionTask.get_parser()
    return parser


def main(cmd=None):
    r"""Target Speaker Extraction model training.

    Example:

        % python enh_tse_train.py asr --print_config --optim adadelta \
                > conf/train_enh.yaml
        % python enh_tse_train.py --config conf/train_enh.yaml
    """
    TargetSpeakerExtractionTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
