import string
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest

from espnet2.bin.st_inference import Speech2Text, get_parser, main
from espnet2.bin.st_inference_streaming import Speech2TextStreaming
from espnet2.legacy.nets.beam_search import Hypothesis
from espnet2.tasks.st import STTask


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
def src_token_list(tmp_path: Path):
    with (tmp_path / "src_tokens.txt").open("w") as f:
        f.write("<blank>\n")
        for c in string.ascii_letters:
            f.write(f"{c}\n")
        f.write("<unk>\n")
        f.write("<sos/eos>\n")
    return tmp_path / "src_tokens.txt"


@pytest.fixture()
def st_config_file(tmp_path: Path, token_list, src_token_list):
    # Write default configuration file
    STTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "st"),
            "--token_list",
            str(token_list),
            "--src_token_list",
            str(src_token_list),
            "--token_type",
            "char",
        ]
    )
    return tmp_path / "st" / "config.yaml"


@pytest.fixture()
def st_config_file_streaming(tmp_path: Path, token_list, src_token_list):
    STTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "st_streaming"),
            "--token_list",
            str(token_list),
            "--src_token_list",
            str(src_token_list),
            "--token_type",
            "char",
            "--frontend",
            "default",
            "--encoder",
            "contextual_block_transformer",
            "--encoder_conf",
            "look_ahead=16",
            "--encoder_conf",
            "hop_size=16",
            "--encoder_conf",
            "block_size=40",
            "--decoder",
            "transformer",
        ]
    )
    return tmp_path / "st_streaming" / "config.yaml"


@pytest.mark.execution_timeout(20)
def test_Speech2TextStreaming_from_pretrained(st_config_file_streaming):
    # model_tag=None skips the model-zoo download and builds from the kwargs,
    # the same path every other from_pretrained in espnet2/bin takes.
    speech2text = Speech2TextStreaming.from_pretrained(
        model_tag=None,
        st_train_config=st_config_file_streaming,
        beam_size=1,
    )
    results = speech2text(np.random.randn(2048), is_final=True)
    for text, token, token_int, hyp in results:
        assert text is None or isinstance(text, str)
        assert isinstance(hyp, Hypothesis)


@pytest.mark.execution_timeout(20)
def test_Speech2TextStreaming_from_pretrained_tag(
    monkeypatch, st_config_file_streaming
):
    class FakeDownloader:
        def download_and_unpack(self, model_tag):
            assert model_tag == "espnet/some_streaming_st"
            return {"st_train_config": str(st_config_file_streaming)}

    monkeypatch.setattr("espnet_model_zoo.downloader.ModelDownloader", FakeDownloader)
    speech2text = Speech2TextStreaming.from_pretrained(
        "espnet/some_streaming_st", beam_size=1
    )
    assert isinstance(speech2text, Speech2TextStreaming)
    assert speech2text.beam_search.beam_size == 1


@pytest.mark.execution_timeout(5)
def test_Speech2Text(st_config_file):
    speech2text = Speech2Text(st_train_config=st_config_file, beam_size=1)
    speech = np.random.randn(1000)
    results = speech2text(speech)
    for text, token, token_int, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)
