from argparse import ArgumentParser

import numpy as np
import pytest

from espnet2.bin.enh_scoring import get_parser, main, scoring
from espnet2.fileio.sound_scp import SoundScpWriter


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


@pytest.fixture
def spk_scp(tmp_path):
    p = tmp_path / "wav.scp"
    w = SoundScpWriter(tmp_path / "data", p)
    w["a"] = 16000, np.random.randint(-100, 100, (160000,), dtype=np.int16)
    w["b"] = 16000, np.random.randint(-100, 100, (80000,), dtype=np.int16)
    return str(p)


@pytest.mark.parametrize("flexible_numspk", [True, False])
@pytest.mark.parametrize("is_tse", [True, False])
def test_scoring(tmp_path, spk_scp, flexible_numspk, is_tse):
    scoring(
        output_dir=str(tmp_path / "output"),
        dtype="float32",
        log_level="INFO",
        key_file=spk_scp,
        ref_scp=[spk_scp],
        inf_scp=[spk_scp],
        ref_channel=0,
        flexible_numspk=flexible_numspk,
        is_tse=is_tse,
        use_dnsmos=False,
        dnsmos_args={
            "mode": "local",
            "auth_key": "",
            "primary_model": "",
            "p808_model": "",
        },
        use_pesq=False,
    )


def test_scoring_pesq_keeps_sample_rate(tmp_path, monkeypatch):
    # Audio above 16 kHz is resampled for PESQ only. The other metrics, and the
    # PESQ of the following speakers and utterances, must still see the original
    # sample rate.
    import sys
    import types

    pesq_calls = []

    class PesqError:
        NO_UTTERANCES_DETECTED = -2
        RETURN_VALUES = 1

    def fake_pesq(fs, ref, deg, mode, on_error=None):
        pesq_calls.append((fs, len(ref), len(deg), mode))
        return 1.0

    monkeypatch.setitem(
        sys.modules,
        "pesq",
        types.SimpleNamespace(PesqError=PesqError, pesq=fake_pesq),
    )
    stoi_rates = []

    def fake_stoi(ref, deg, fs_sig, extended=False):
        stoi_rates.append(fs_sig)
        return 0.5

    monkeypatch.setattr("espnet2.bin.enh_scoring.stoi", fake_stoi)

    def fake_bss_eval_sources(ref, inf, compute_permutation=True):
        # not what this test is about, and it takes seconds on this much audio
        zeros = np.zeros(len(ref))
        return zeros, zeros, zeros, np.arange(len(ref))

    monkeypatch.setattr(
        "espnet2.bin.enh_scoring.bss_eval_sources", fake_bss_eval_sources
    )

    scps = []
    for spk in range(2):
        p = tmp_path / f"spk{spk}.scp"
        w = SoundScpWriter(tmp_path / f"data{spk}", p)
        for key in ("a", "b"):
            w[key] = 48000, np.random.randint(-100, 100, (24000,), dtype=np.int16)
        w.close()
        scps.append(str(p))
    scoring(
        output_dir=str(tmp_path / "output"),
        dtype="float32",
        log_level="INFO",
        key_file=scps[0],
        ref_scp=scps,
        inf_scp=scps,
        ref_channel=0,
        flexible_numspk=False,
        is_tse=True,
        use_dnsmos=False,
        dnsmos_args={},
        use_pesq=True,
    )

    # 2 utterances x 2 speakers
    assert pesq_calls == [(16000, 8000, 8000, "wb")] * 4
    assert stoi_rates == [48000] * 8
