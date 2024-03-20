from argparse import ArgumentParser
from pathlib import Path

import pytest

from espnet2.bin.lm_train import get_parser, main
from espnet2.tasks.lm import LMTask


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


@pytest.fixture()
def token_list(tmp_path: Path):
    with (tmp_path / "tokens.txt").open("w") as f:
        tokens = [
            "<blank>",
            "<unk>",
            "a",
            "i",
            "<sos/eos>",
            "<generatetext>",
            "<generatespeech>",
        ]
        for tok in tokens:
            f.write(f"{tok}\n")
    return tmp_path / "tokens.txt"


@pytest.fixture()
def lm_multitask_config_file(tmp_path: Path, token_list):
    # Write default configuration file
    LMTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "lm"),
            "--token_list",
            str(token_list),
            "--model",
            "lm_multitask",
        ]
    )
    return tmp_path / "lm" / "config.yaml"


def test_main(lm_multitask_config_file):
    with pytest.raises(SystemExit):
        main("--config " + str(lm_multitask_config_file))
