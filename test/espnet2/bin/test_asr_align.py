from argparse import ArgumentParser
from pathlib import Path
import string

import numpy as np
import pytest

from espnet2.bin.asr_align import get_parser
from espnet2.bin.asr_align import main
from espnet2.bin.asr_align import CTCSegmentation
from espnet2.bin.asr_align import CTCSegmentationResult
from espnet2.tasks.asr import ASRTask


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


@pytest.fixture()
def token_list(tmp_path: Path):
    with (tmp_path / "tokens.txt").open("w") as f:
        f.write("<blank>\n")
        for c in string.ascii_letters:
            f.write(f"{c}\n")
        f.write("<unk>\n")
        f.write("<sos/eos>\n")
    return tmp_path / "tokens.txt"


@pytest.fixture()
def asr_config_file(tmp_path: Path, token_list):
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
        ]
    )
    return tmp_path / "asr" / "config.yaml"


@pytest.mark.execution_timeout(5)
def test_CTCSegmentation(asr_config_file):
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
    assert isinstance(segments, CTCSegmentationResult)
    kaldi_text = str(segments)
    first_line = kaldi_text.splitlines()[0]
    assert "utt_a" == first_line.split(" ")[0]
    start, end, score = segments.segments[0]
    assert start > 0.0
    assert start < (num_samples / fs)
    assert end >= start
    assert score < 0.0
    # check options
    option_dict = {
        "subsampling_factor": 512,
        "frame_duration": 10,
        "min_window_size": 100,
        "max_window_size": 20000,
        "set_blank": 0,
        "scoring_length": 10,
        "replace_spaces_with_blanks": True,
        "gratis_blank": True,
    }
    aligner.set_config(**option_dict)
    assert aligner.warned_about_misconfiguration
    aligner.kaldi_style_text = False
    text = ["HOTELS", "HOLIDAY'S STRATEGY", "ASSETS", "PROPERTY MANAGEMENT"]
    segments = aligner(speech, text, name="foo")
    kaldi_segments = str(segments)
    first_line = kaldi_segments.splitlines()[0]
    assert "foo_0000" == first_line.split(" ")[0]
