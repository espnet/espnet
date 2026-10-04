"""Legacy entry point; new recipes use audio_metric_train."""

from espnet2.bin.audio_metric_train import get_parser, main  # noqa: F401

if __name__ == "__main__":
    main()
