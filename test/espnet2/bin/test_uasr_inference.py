import string
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest
import yaml

from espnet2.bin.uasr_inference import Speech2Text, get_parser, main
from espnet2.tasks.uasr import UASRTask
from espnet.nets.beam_search import Hypothesis

def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()

    
@pytest.fixture()
def token_list(tmp_path: Path):
    with (tmp_path / "tokens.txt").open("w") as f:
        f.write("<eps>\n")
        f.write("<s>\n")
        f.write("<pad>\n")
        f.write("</s>\n")
        f.write("<unk>\n")
        f.write("<SIL>\n")
    return tmp_path / "tokens.txt"


@pytest.fixture()
def uasr_config_file(tmp_path: Path, token_list):
    # Write default configuration file
    UASRTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "uasr"),
            "--token_list",
            str(token_list),
            "--token_type",
            "phn",
            "--segmenter",
            "join",
            "--discriminator",
            "conv",
            "--generator",
            "conv",
            "--write_collected_feats",
            "false",
            "--input_size",
            "512",
        ]
    )
    return tmp_path / "uasr" / "config.yaml"


@pytest.mark.execution_timeout(5)
def test_Speech2Text(uasr_config_file):
    speech2text = Speech2Text(
        uasr_train_config=uasr_config_file, beam_size=1
    )
    speech = np.random.randn(100, 512)
    results = speech2text(speech)
    for text, token, token_int, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)