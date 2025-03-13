from argparse import ArgumentParser
from pathlib import Path

import k2
import numpy as np
import pytest
import torch

from espnet2.bin.uasr_inference_k2 import get_parser, k2Speech2Text, main
from espnet2.tasks.uasr import UASRTask


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
def token_list_file(tmp_path: Path):
    with (tmp_path / "tokens_list_file").open("w") as f:
        f.write("<eps> 0\n")
        f.write("<s> 1\n")
        f.write("<pad> 2\n")
        f.write("</s> 3\n")
        f.write("<unk> 4\n")
        f.write("<SIL> 5\n")
    return str(tmp_path / "tokens_list_file")


@pytest.fixture()
def decoding_graph(tmp_path: Path):
    num_tokens = 6
    start_state = 0
    next_state = start_state
    final_state = 1

    arcs = ""
    for i in range(num_tokens):
        arcs += f"{start_state} {next_state} {i} {i} 0\n"
    arcs += f"{next_state} {final_state} -1 -1 0.0\n"
    arcs += f"{final_state}"
    graph = k2.Fsa.from_str(arcs, num_aux_labels=1)
    graph = k2.arc_sort(graph)
    torch.save(graph.as_dict(), tmp_path / "HLG.pt")
    return str(tmp_path / "HLG.pt")


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
def test_Speech2Text(uasr_config_file, decoding_graph, token_list_file):
    speech2text = k2Speech2Text(
        uasr_train_config=uasr_config_file,
        decoding_graph=decoding_graph,
        beam_size=1,
        token_list_file=token_list_file,
        token_type="word",
    )
    speech = np.random.randn(100, 512)
    results = speech2text(speech)
    for text, token, token_int, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, float)
