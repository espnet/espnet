import string
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest

from espnet2.bin.s2t_align import CTCSegmentation, CTCSegmentationTask, get_parser, main
from espnet2.tasks.s2t_ctc import S2TTask


def test_get_parser():
    """Check the parser."""
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    """Run main(·) once."""
    with pytest.raises(SystemExit):
        main()


@pytest.fixture()
def token_list(tmp_path: Path):
    with (tmp_path / "tokens.txt").open("w") as f:
        tokens = [
            "<blank>",
            "<unk>",
            "<na>",
            "<nolang>",
            "<eng>",
            "<zho>",
            "<asr>",
            "<st_eng>",
            *list(string.ascii_letters),
            "<sos>",
            "<eos>",
            "<sop>",
        ]
        for tok in tokens:
            f.write(f"{tok}\n")
    return tmp_path / "tokens.txt"


@pytest.fixture()
def s2t_config_file(tmp_path: Path, token_list):
    # Write default configuration file
    S2TTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "s2t"),
            "--token_list",
            str(token_list),
            "--token_type",
            "char",
            "--promptencoder_conf",
            "output_size=4",
            "--preprocessor_conf",
            "fs=16000",
            "--preprocessor_conf",
            "speech_length=4",
            "--frontend_conf",
            "fs=16k",
            "--frontend_conf",
            "hop_length=160",
            "--encoder_conf",
            "input_layer=conv2d8",
        ]
    )
    return tmp_path / "s2t" / "config.yaml"


@pytest.mark.execution_timeout(5)
def test_CTCSegmentation(s2t_config_file):
    """Test CTC segmentation.

    Note that due to the random vector that is given to the CTC segmentation function,
    there is a small chance that this test might randomly fail. If this ever happens,
    use the test file test_utils/ctc_align_test.wav instead, or a fixed test vector.
    """

    num_samples = 200000
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
        s2t_train_config=s2t_config_file,
        fs=fs,
        context_len_in_secs=1,
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


def test_the_old_module_name_still_aligns(s2t_config_file):
    """`espnet2.bin.s2t_ctc_align` was this module until 202610.

    The recipes in this repository import the new name, but a script
    someone else wrote does not, so the old path forwards and says so.
    """
    from espnet2.bin.s2t_ctc_align import CTCSegmentation as Moved
    from espnet2.bin.s2t_ctc_align import CTCSegmentationTask as MovedTask

    assert issubclass(Moved, CTCSegmentation)
    assert MovedTask is CTCSegmentationTask

    with pytest.warns(DeprecationWarning, match="espnet2.bin.s2t_align"):
        aligner = Moved(
            s2t_train_config=s2t_config_file,
            fs=16000,
            context_len_in_secs=1,
            kaldi_style_text=True,
            min_window_size=10,
        )

    speech = np.random.randn(200000)
    segments = aligner(speech, "utt_a HOTELS\nutt_b ASSETS\n", fs=16000)
    assert isinstance(segments, CTCSegmentationTask)
    assert str(segments).splitlines()[0].split(" ")[0] == "utt_a"


def test_the_old_script_path_still_runs():
    """`python espnet2/bin/s2t_ctc_align.py` keeps working, and says so."""
    from espnet2.bin import s2t_ctc_align

    parser = s2t_ctc_align.get_parser()
    assert isinstance(parser, ArgumentParser)
    assert "moved" in parser.description
    # the arguments are the new module's, so a recipe calling the script
    # does not have to change either
    options = {a.option_strings[0] for a in parser._actions}
    assert "--s2t_train_config" in options

    with pytest.raises(SystemExit):
        with pytest.warns(DeprecationWarning, match="espnet2.bin.s2t_align"):
            s2t_ctc_align.main()
