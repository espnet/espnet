#!/usr/bin/env python3

from espnet2.tasks.spk import SpeakerTask


def get_parser():
    parser = Speaker.get_parser()
    return parser


def main(cmd=None):
    r"""Speaker diarization training.

    Example:
        % python diar_train.py diar --print_config --optim adadelta \
                > conf/train_diar.yaml
        % python diar_train.py --config conf/train_diar.yaml
    """
    SpeakerTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
