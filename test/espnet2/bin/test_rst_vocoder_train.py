from argparse import ArgumentParser

import pytest

from espnet2.bin.rst_vocoder_train import get_parser, main


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_vocoder_type_choices():
    parser = get_parser()
    args = parser.parse_args(["--vocoder_type", "hifigan", "--output_dir", "x"])
    assert args.vocoder_type == "hifigan"
    with pytest.raises(SystemExit):
        parser.parse_args(["--vocoder_type", "unknown", "--output_dir", "x"])


def test_main():
    with pytest.raises(SystemExit):
        main()
