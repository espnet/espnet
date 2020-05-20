from argparse import ArgumentParser

import pytest

from espnet2.bin.aggregate_stats_dirs import get_parser
from espnet2.bin.aggregate_stats_dirs import main


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()
