#!/usr/bin/env python3

from espnet2.tasks.spk import SpeakerTask


def get_parser():
    parser = SpeakerTask.get_parser()
    return parser


def main(cmd=None):
    r"""Speaker embedding extractor training.

    Trained model can be used for
    speaker verification, open set speaker identification, and also as
    embeddings for various other tasks including speaker diarization.

    Example:
        % python spk_train.py --print_config --optim adadelta \
                > conf/train_spk.yaml
        % python spk_train.py --config conf/train_diar.yaml
    """
    SpeakerTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
