"""Compatibility entry point; new recipes use aqa_train."""

from espnet2.bin.aqa_train import get_parser, main  # noqa: F401

if __name__ == "__main__":
    main()
