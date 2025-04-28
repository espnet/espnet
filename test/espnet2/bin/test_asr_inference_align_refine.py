import string
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest
import yaml

from espnet2.bin.asr_inference_align_refine import Speech2Text, get_parser, main
from espnet2.tasks.asr import ASRTask
from espnet2.utils.yaml_no_alias_safe_dump import yaml_no_alias_safe_dump
from espnet.nets.beam_search import Hypothesis


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
def asr_config_file(tmp_path: Path, token_list):
    # Write default configuration file
    ASRTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "asr"),
            "--token_list",
            str(token_list),
            "--model",
            "align_refine_model",
            "--token_type",
            "char",
            "--decoder",
            "transformer",
        ]
    )

    # overwrite decoder conf
    with open(tmp_path / "asr" / "config.yaml", "r") as f:
        args = yaml.safe_load(f)
    args.update({"decoder_conf": {"use_output_layer": False}})
    with open(tmp_path / "asr" / "config.yaml", "w") as f:
        yaml_no_alias_safe_dump(args, f, indent=4, sort_keys=False)

    return tmp_path / "asr" / "config.yaml"


@pytest.mark.execution_timeout(20)
@pytest.mark.parametrize("align_refine_k", [1, 4])
@pytest.mark.parametrize("ctc_only", [True, False])
def test_Speech2Text(asr_config_file, align_refine_k, ctc_only):
    speech2text = Speech2Text(
        asr_train_config=asr_config_file,
        beam_size=1,
        align_refine_k=align_refine_k,
        ctc_only=ctc_only,
    )
    speech = np.random.randn(100000)
    results = speech2text(speech)
    for text, token, token_int, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)
