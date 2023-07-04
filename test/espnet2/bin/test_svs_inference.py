import string
from argparse import ArgumentParser
from pathlib import Path

import pytest

from espnet2.bin.svs_inference import SingingGenerate, get_parser, main
from espnet2.tasks.svs import SVSTask


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


@pytest.fixture()
def token_list(tmp_path: Path):
    with (tmp_path / "tokens.txt").open("w") as f:
        f.write("<blank>\n")
        for c in string.ascii_letters:
            f.write(f"{c}\n")
        f.write("<unk>\n")
        f.write("<sos/eos>\n")
    return tmp_path / "tokens.txt"


@pytest.fixture()
def config_file(tmp_path: Path, token_list):
    # Write default configuration file
    SVSTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path),
            "--token_list",
            str(token_list),
            "--token_type",
            "char",
            "--cleaner",
            "none",
            "--g2p",
            "none",
            "--normalize",
            "none",
            "--feats_extract_conf",
            "hop_length=256",
        ]
    )
    return tmp_path / "config.yaml"


@pytest.mark.execution_timeout(10)
def test_SingingGenerate(config_file):
    svs = SingingGenerate(train_config=config_file)

    phn_dur = [
        [0.0, 0.219],
        [0.219, 0.50599998],
        [0.50599998, 0.71399999],
        [0.71399999, 1.097],
        [1.097, 1.28799999],
        [1.28799999, 1.98300004],
    ]
    phn = ["sh", "i", "q", "v", "n", "i"]
    score = [
        [0, 0.50625, "sh_i", 58, "sh_i"],
        [0.50625, 1.09728, "q_v", 56, "q_v"],
        [1.09728, 1.9832100000000001, "n_i", 53, "n_i"],
    ]
    tempo = 70
    tmp = {}
    tmp["label"] = phn_dur, phn
    tmp["score"] = tempo, score
    default_inp = tmp

    svs(text=default_inp)
