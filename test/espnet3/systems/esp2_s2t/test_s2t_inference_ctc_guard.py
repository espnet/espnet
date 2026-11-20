"""The timestamp options refused on a CTC-only checkpoint.

Both need a decoder to ride on, and a CTC-only checkpoint has none. The
fixtures here build such a checkpoint, which is why this lives beside
``test_s2t_inference.py`` rather than in it: that file's fixtures of the same
name build a checkpoint that does have a decoder.
"""

from pathlib import Path

import pytest

from espnet2.tasks.s2t_ctc import S2TTask
from espnet3.systems.esp2_s2t.src.s2t_inference import Speech2Text

pytest.importorskip("whisper")


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
            "a",
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
            "fs=2000",
            "--preprocessor_conf",
            "speech_length=3",
            "--frontend_conf",
            "fs=16k",
            "--frontend_conf",
            "hop_length=160",
            "--encoder_conf",
            "input_layer=conv2d8",
        ]
    )
    return tmp_path / "s2t" / "config.yaml"


@pytest.mark.execution_timeout(10)
@pytest.mark.parametrize(
    "option", [{"speaker_change_symbol": "a"}, {"adaptive_timestamp": True}]
)
def test_the_base_class_refuses_timestamp_options_on_a_ctc_only_checkpoint(
    s2t_config_file, option
):
    """Both options need a decoder, and a CTC-only checkpoint has none.

    Without the guard the request is not refused but lost. The filter is
    built and then dropped, because the CTC-only branch gives it no weight
    and BeamSearch keeps only the scorers that have one, so a user who asked
    for a speaker change symbol would get an unconstrained decode and no
    warning. adaptive_timestamp does not even get that far: it reads a
    decoder that is not there.
    """
    with pytest.raises(ValueError, match="CTC-only"):
        Speech2Text(s2t_train_config=s2t_config_file, beam_size=1, **option)
