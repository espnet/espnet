"""egs2/TEMPLATE/asr1/pyscripts/utils/make_windows.py (diar.sh --train_window_size)."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from espnet2.fileio.rttm import RttmReader
from espnet2.fileio.segments import seconds_to_sample
from espnet2.tasks.diar import SpeakerLabelPadder

SCRIPT = (
    Path(__file__).resolve().parents[3]
    / "egs2/TEMPLATE/asr1/pyscripts/utils/make_windows.py"
)


@pytest.fixture(scope="module")
def make_windows():
    spec = importlib.util.spec_from_file_location("make_windows", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules["make_windows"] = module
    spec.loader.exec_module(module)
    yield module.make_windows
    sys.modules.pop("make_windows", None)


def _write_src(src, recordings):
    """recordings: rec -> (num_samples, [(spk, start, end inclusive), ...])."""
    src.mkdir()
    with (
        (src / "wav.scp").open("w") as wav,
        (src / "utt2num_samples").open("w") as n,
        (src / "espnet_rttm").open("w") as rttm,
    ):
        for rec, (num, turns) in recordings.items():
            wav.write(f"{rec} /dev/null:0:0\n")
            n.write(f"{rec} {num}\n")
            for spk, s, e in turns:
                rttm.write(f"SPEAKER {rec} 1 {s} {e} <NA> <NA> {spk} <NA>\n")
            rttm.write(f"END {rec} <NA> <NA> {num} <NA> <NA> <NA> <NA>\n")


RECORDINGS = {
    # two speakers, overlapping turns of spk1, a silent stretch 4000-5999
    "recA": (
        10500,
        [
            ("spk1", 100, 1999),
            ("spk1", 1500, 2599),
            ("spk2", 2400, 3999),
            ("spk2", 6000, 10499),
        ],
    ),
    # one speaker
    "recB": (4000, [("spk3", 0, 999)]),
    # no speech at all
    "recC": (3000, []),
}


def _window_labels(dst, num_spk):
    reader = RttmReader(str(dst / "espnet_rttm"))
    pad = SpeakerLabelPadder(lambda uid, data: data, num_spk)
    return {k: pad(k, {"spk_labels": reader[k]})["spk_labels"] for k in reader}


@pytest.mark.parametrize("window, shift", [(2000, 2000), (2000, 1000), (3000, 3000)])
def test_windows_cut_the_recording_labels(make_windows, tmp_path, window, shift):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_src(src, RECORDINGS)
    n_rec, n_win, n_empty = make_windows(src, dst, window, shift, 16000)

    whole = RttmReader(str(src / "espnet_rttm"))
    labels = _window_labels(dst, num_spk=2)
    segments = [line.split() for line in (dst / "segments").read_text().splitlines()]
    expected = sum(
        len(range(0, num - window + 1, shift)) for num, _ in RECORDINGS.values()
    )
    assert n_rec == 3 and n_win == len(segments) == expected
    empty = 0
    for wid, rec, start, end in segments:
        off = seconds_to_sample(float(start), 16000)
        assert seconds_to_sample(float(end), 16000) == off + window
        assert wid == f"{rec}-{off:010d}"
        label = labels[wid]
        assert label.shape == (window, 2)
        # the recording's label cut to the window, speakers in any column order
        if rec in whole:
            ref = whole[rec][off : off + window]
            ref = np.pad(ref, ((0, 0), (0, 2 - ref.shape[1])))
        else:
            ref = np.zeros((window, 2))
        got_cols = sorted(map(tuple, label.T.astype(int)))
        ref_cols = sorted(map(tuple, ref.T.astype(int)))
        assert got_cols == ref_cols
        empty += not label.any()
    assert empty == n_empty
    assert (dst / "wav.scp").read_text() == (src / "wav.scp").read_text()


def test_tail_shorter_than_a_window_is_dropped(make_windows, tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_src(src, {"rec": (4999, [("s", 0, 4998)])})
    _, n_win, _ = make_windows(src, dst, 2000, 2000, 16000)
    assert n_win == 2  # windows at 0 and 2000; 4000-4998 is shorter than 2000


def test_window_without_speech_has_only_its_end_line(make_windows, tmp_path):
    src, dst = tmp_path / "src", tmp_path / "dst"
    _write_src(src, {"rec": (4000, [("s", 2500, 3999)])})
    _, n_win, n_empty = make_windows(src, dst, 2000, 2000, 16000)
    assert (n_win, n_empty) == (2, 1)
    lines = (dst / "espnet_rttm").read_text().splitlines()
    assert lines[0] == "END rec-0000000000 <NA> <NA> 2000 <NA> <NA> <NA> <NA>"
    assert lines[1] == "SPEAKER rec-0000002000 1 500 1999 <NA> <NA> s <NA>"
    labels = _window_labels(dst, num_spk=1)
    assert labels["rec-0000000000"].shape == (2000, 1)
    assert not labels["rec-0000000000"].any()
    assert (
        labels["rec-0000002000"][500:].all()
        and not labels["rec-0000002000"][:500].any()
    )


def test_bad_window(make_windows, tmp_path):
    src = tmp_path / "src"
    _write_src(src, RECORDINGS)
    with pytest.raises(ValueError):
        make_windows(src, tmp_path / "dst", 0, 1, 16000)
