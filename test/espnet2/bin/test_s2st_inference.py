import string
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest
import torch

from espnet2.bin.s2st_inference import Speech2Speech, get_parser, main
from espnet2.tasks.s2st import S2STTask


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


@pytest.fixture()
def tgt_token_list(tmp_path: Path):
    with (tmp_path / "tgt_tokens.txt").open("w") as f:
        f.write("<blank>\n")
        for c in string.ascii_letters:
            f.write(f"{c}\n")
        f.write("<unk>\n")
        f.write("<sos/eos>\n")
    return tmp_path / "tgt_tokens.txt"


@pytest.fixture()
def src_token_list(tmp_path: Path):
    with (tmp_path / "src_tokens.txt").open("w") as f:
        f.write("<blank>\n")
        for c in string.ascii_letters:
            f.write(f"{c}\n")
        f.write("<unk>\n")
        f.write("<sos/eos>\n")
    return tmp_path / "src_tokens.txt"


@pytest.fixture()
def unit_token_list(tmp_path: Path):
    with (tmp_path / "unit_tokens.txt").open("w") as f:
        f.write("<blank>\n")
        for c in range(10):
            f.write(f"{c}\n")
        f.write("<unk>\n")
        f.write("<sos/eos>\n")
    return tmp_path / "unit_tokens.txt"


@pytest.fixture()
def s2st_config_file(tmp_path: Path, tgt_token_list, src_token_list, unit_token_list):
    # Write default configuration file
    S2STTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "s2st"),
            "--tgt_token_list",
            str(tgt_token_list),
            "--src_token_list",
            str(src_token_list),
            "--unit_token_list",
            str(unit_token_list),
            "--tgt_token_type",
            "char",
            "--src_token_type",
            "char",
            "--input_size",
            "20",
        ]
    )
    return tmp_path / "s2st" / "config.yaml"


@pytest.mark.execution_timeout(5)
def test_Speech2Speech(s2st_config_file):
    speech2speech = Speech2Speech(train_config=s2st_config_file, beam_size=1)
    speech = np.random.randn(1, 10, 20)
    results = speech2speech(speech)
    assert isinstance(results, dict)
    assert "feat_gen" in results.keys()
    assert isinstance(results["feat_gen"], torch.Tensor)
