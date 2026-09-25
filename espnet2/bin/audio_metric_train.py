#!/usr/bin/env python3
from espnet2.tasks.audio_metric import AudioMetricTask


def get_parser():
    """Return the audio metric training argument parser."""
    parser = AudioMetricTask.get_parser()
    return parser


def main(cmd=None):
    """Train an audio metric predictor.

    Example:

        % python audio_metric_train.py --print_config --optim adadelta
        % python audio_metric_train.py --config conf/train_universa.yaml
    """
    AudioMetricTask.main(cmd=cmd)


if __name__ == "__main__":
    main()
