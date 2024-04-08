"""Tests for asr_align.py."""

import string
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest

from espnet2.bin.asr_align import CTCSegmentation, CTCSegmentationTask, get_parser, main
from espnet2.tasks.asr import ASRTask


def test_get_parser():
    """Check the parser."""
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    """Run main(·) once."""
    with pytest.raises(SystemExit):
        main()


@pytest.fixture()
def token_list(tmp_path: Path):
    """Obtain a test file with a list of tokens."""
    with (tmp_path / "tokens.txt").open("w") as f:
        f.write("<blank>\n")
        for c in string.ascii_letters:
            f.write(f"{c}\n")
        f.write("<unk>\n")
        f.write("<sos/eos>\n")
    return tmp_path / "tokens.txt"


@pytest.fixture()
def asr_config_file(tmp_path: Path, token_list):
    """Obtain ASR config file for test."""
    # Write default configuration file
    ASRTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "asr"),
            "--token_list",
            str(token_list),
            "--token_type",
            "char",
            "--decoder",
            "rnn",
        ]
    )
    return tmp_path / "asr" / "config.yaml"


@pytest.mark.execution_timeout(5)
def test_CTCSegmentation(asr_config_file):
    """Test CTC segmentation.

    Note that due to the random vector that is given to the CTC segmentation function,
    there is a small chance that this test might randomly fail. If this ever happens,
    use the test file test_utils/ctc_align_test.wav instead, or a fixed test vector.
    """

    num_samples = 100000
    fs = 16000
    # text includes:
    #   one blank line
    #   kaldi-style utterance names
    #   one char not included in char list
    text = (
        "\n"
        "utt_a HOTELS\n"
        "utt_b HOLIDAY'S STRATEGY\n"
        "utt_c ASSETS\n"
        "utt_d PROPERTY MANAGEMENT\n"
    )
    # speech either from the test audio file or random
    speech = np.random.randn(num_samples)
    aligner = CTCSegmentation(
        asr_train_config=asr_config_file,
        fs=fs,
        kaldi_style_text=True,
        min_window_size=10,
    )
    segments = aligner(speech, text, fs=fs)
    # check segments
    assert isinstance(segments, CTCSegmentationTask)
    kaldi_text = str(segments)
    first_line = kaldi_text.splitlines()[0]
    assert "utt_a" == first_line.split(" ")[0]
    start, end, score = segments.segments[0]
    assert start > 0.0
    assert start < (num_samples / fs)
    assert end >= start
    assert score < 0.0
    # check options and align with "classic" text converter
    option_dict = {
        "fs": 16000,
        "time_stamps": "fixed",
        "samples_to_frames_ratio": 512,
        "min_window_size": 100,
        "max_window_size": 20000,
        "set_blank": 0,
        "scoring_length": 10,
        "replace_spaces_with_blanks": True,
        "gratis_blank": True,
        "kaldi_style_text": False,
        "text_converter": "classic",
    }
    aligner.set_config(**option_dict)
    assert aligner.warned_about_misconfiguration
    text = ["HOTELS", "HOLIDAY'S STRATEGY", "ASSETS", "PROPERTY MANAGEMENT"]
    segments = aligner(speech, text, name="foo")
    segments_str = str(segments)
    first_line = segments_str.splitlines()[0]
    assert "foo_0000" == first_line.split(" ")[0]
    # test the ratio estimation (result: 509)
    ratio = aligner.estimate_samples_to_frames_ratio()
    assert 500 <= ratio <= 520
