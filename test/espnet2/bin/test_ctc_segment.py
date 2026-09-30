"""Tests for ctc_segment.py, the algorithm the two align scripts share."""

import pytest

from espnet2.bin.asr_align import CTCSegmentation as AsrCTCSegmentation
from espnet2.bin.ctc_segment import AbsCTCSegmentation, build_parser
from espnet2.bin.s2t_align import CTCSegmentation as S2TCTCSegmentation


def test_each_model_has_its_own_segmentation_parameters():
    """`set_config` writes into `config`, so it must not be one shared object.

    The two classes had one each while they were two copies of the module;
    inheriting one from the base would have a caller aligning with both
    models change the parameters of the one by configuring the other.
    """
    assert AsrCTCSegmentation.config is not S2TCTCSegmentation.config
    assert AsrCTCSegmentation.config is not AbsCTCSegmentation.config


@pytest.mark.parametrize(
    "prefix, expected",
    [("asr", "--asr_train_config"), ("s2t", "--s2t_train_config")],
)
def test_each_script_keeps_its_own_model_arguments(prefix, expected):
    """The scripting interface is what recipes call; its names do not move."""
    parser = build_parser(prefix, "a description")
    options = {action.option_strings[0] for action in parser._actions}
    assert expected in options
    assert f"--{prefix}_model_file" in options


@pytest.mark.parametrize(
    "device, ngpu, expected",
    [(None, 0, "cpu"), (None, 1, "cuda"), ("mps", 0, "mps"), ("cuda:1", 1, "cuda:1")],
)
def test_device_is_taken_over_ngpu(device, ngpu, expected):
    assert AbsCTCSegmentation._resolve_device(device, ngpu) == expected


def test_more_than_one_gpu_is_refused():
    with pytest.raises(NotImplementedError):
        AbsCTCSegmentation._resolve_device(None, 2)
