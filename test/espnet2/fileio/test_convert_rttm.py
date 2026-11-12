"""egs2/TEMPLATE/asr1/pyscripts/utils/convert_rttm.py (the diar1 stage 2/3 step)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "egs2/TEMPLATE/asr1/pyscripts/utils/convert_rttm.py"
)


@pytest.fixture(scope="module")
def convert_rttm_text():
    spec = importlib.util.spec_from_file_location("convert_rttm", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["convert_rttm"] = module  # typeguard looks the module up
    spec.loader.exec_module(module)
    yield module.convert_rttm_text
    sys.modules.pop("convert_rttm", None)


def _write_inputs(tmp_path, wav_entry):
    (tmp_path / "rttm").write_text("SPEAKER rec1 1 0.50 1.00 <NA> <NA> spk1 <NA>\n")
    (tmp_path / "wav.scp").write_text(f"rec1 {wav_entry}\n")


def test_lengths_from_audio(tmp_path, convert_rttm_text):
    wav = tmp_path / "rec1.wav"
    soundfile.write(wav, np.zeros(32000), 16000)
    _write_inputs(tmp_path, wav)
    convert_rttm_text(tmp_path / "rttm", tmp_path / "wav.scp", 16000, tmp_path)
    assert (tmp_path / "espnet_rttm").read_text().splitlines() == [
        "SPEAKER rec1 1 8000 24000 <NA> <NA> spk1 <NA>",
        "END rec1 <NA> <NA> 32000 <NA> <NA> <NA> <NA>",
    ]


def test_lengths_from_utt2num_samples(tmp_path, convert_rttm_text):
    # an omniio entry cannot be opened by soundfile; the length must come from
    # utt2num_samples, which format_wav_scp.sh writes for every audio format
    _write_inputs(tmp_path, "/no/such/blob_0.bin:0:1234")
    (tmp_path / "utt2num_samples").write_text("rec1 32000\n")
    convert_rttm_text(
        tmp_path / "rttm",
        tmp_path / "wav.scp",
        16000,
        tmp_path,
        utt2num_samples_path=tmp_path / "utt2num_samples",
    )
    assert (tmp_path / "espnet_rttm").read_text().splitlines()[-1] == (
        "END rec1 <NA> <NA> 32000 <NA> <NA> <NA> <NA>"
    )
