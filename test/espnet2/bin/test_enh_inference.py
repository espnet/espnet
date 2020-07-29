from argparse import ArgumentParser

import pytest

from espnet2.bin.enh_inference import get_parser
from espnet2.bin.enh_inference import main


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()
