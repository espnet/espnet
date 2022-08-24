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
def test_scoring(tmp_path, spk_scp, flexible_numspk):
    scoring(
        output_dir=str(tmp_path / "output"),
        dtype="float32",
        log_level="INFO",
        key_file=spk_scp,
        ref_scp=[spk_scp],
        inf_scp=[spk_scp],
        ref_channel=0,
        flexible_numspk=flexible_numspk,
    )
