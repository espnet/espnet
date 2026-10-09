import string
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest
import torch

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
        context_len_in_secs=0.8,
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
            context_len_in_secs=0.8,
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


@pytest.mark.parametrize("batch_size", [1, 3])
@pytest.mark.parametrize("context", [0.56, 0.8, 1.6])
def test_get_lpz_keeps_contiguous_audio_frames(
    s2t_config_file, monkeypatch, batch_size, context
):
    """Buffer joins keep their source time even when the encoder loses an edge."""
    aligner = CTCSegmentation(
        s2t_train_config=s2t_config_file,
        fs=16000,
        context_len_in_secs=context,
        batch_size=batch_size,
    )
    hop = aligner.samples_to_frames_ratio

    def encode(speech, prefix, **kwargs):
        frames = speech[:, ::hop][:, :-1].unsqueeze(-1)
        marks = speech.new_full((speech.size(0), prefix.size(1), 1), -1.0)
        return torch.cat([marks, frames], dim=1), None

    monkeypatch.setattr(aligner.s2t_model, "encode", encode)
    monkeypatch.setattr(aligner.ctc, "log_softmax", lambda enc: enc)
    speech = np.arange(1, 10 * aligner.fs + 1, dtype=np.float32)
    probs, covered_samples = aligner.get_lpz(speech)
    wanted = round(len(speech) / hop)
    assert probs[:wanted, 0].tolist() == (1 + hop * np.arange(wanted)).tolist()
    assert len(probs) * hop == covered_samples


@pytest.mark.parametrize(
    "context,window,argument",
    [
        (0.5, 4, "context_len_in_secs"),
        (0.8, 3, "speech_length"),
    ],
)
def test_get_lpz_rejects_fractional_frames(s2t_config_file, context, window, argument):
    """Invalid frame grids fail before encoding or aligning the recording."""
    aligner = CTCSegmentation(
        s2t_train_config=s2t_config_file,
        fs=16000,
        context_len_in_secs=context,
    )
    aligner.s2t_train_args.preprocessor_conf["speech_length"] = window
    with pytest.raises(ValueError, match=argument + ".*whole number"):
        aligner.get_lpz(np.zeros(16000, dtype=np.float32))


@pytest.mark.parametrize("context", [-0.8, 0, 2, 2.4])
def test_get_lpz_rejects_missing_chunk_frames(s2t_config_file, context):
    """Refuse empty steps and contexts too short to preserve complete chunks."""
    aligner = CTCSegmentation(
        s2t_train_config=s2t_config_file,
        fs=16000,
        context_len_in_secs=context,
    )
    with pytest.raises(ValueError, match="context"):
        aligner.get_lpz(np.zeros(16000, dtype=np.float32))
