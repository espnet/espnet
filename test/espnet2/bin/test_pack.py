from argparse import ArgumentParser
from inspect import signature

import pytest

from espnet2.bin.cls_inference import Classification
from espnet2.bin.pack import ClassificationPackedContents, get_parser, main


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


def test_cls_packing():
    packing_obj = ClassificationPackedContents()
    valid_args = signature(Classification.__init__).parameters
    for filename in packing_obj.files:
        assert filename in valid_args, f"{filename} not in {valid_args}"
    for yaml_filename in packing_obj.yaml_files:
        assert yaml_filename in valid_args, f"{yaml_filename} not in {valid_args}"
