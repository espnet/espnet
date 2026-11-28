import string
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

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


@pytest.mark.parametrize("speech_length", [16000, 16050])
@pytest.mark.parametrize("chunk", [128, 205, 410, 640, 1000, 1600])
@pytest.mark.parametrize(
    "n_fft,win_length,hop_length", [(512, 512, 128), (512, 400, 160), (256, 256, 128)]
)
def test_Speech2TextStreaming_frontend(
    st_config_file_streaming, chunk, n_fft, win_length, hop_length, speech_length
):
    config = yaml.safe_load(st_config_file_streaming.read_text())
    config["frontend_conf"] = dict(
        n_fft=n_fft, win_length=win_length, hop_length=hop_length
    )
    config["normalize"] = None
    st_config_file_streaming.write_text(yaml.safe_dump(config))
    speech2text = Speech2TextStreaming(
        st_train_config=st_config_file_streaming, beam_size=1
    )
    speech = torch.randn(speech_length, generator=torch.Generator().manual_seed(0))
    with torch.no_grad():
        offline, _ = speech2text.st_model._extract_feats(
            speech[None], torch.tensor([speech.numel()])
        )
        feats, states = [], None
        for start in range(0, speech.numel(), chunk):
            feat, _, states = speech2text.apply_frontend(
                speech[start : start + chunk],
                states,
                is_final=start + chunk >= speech.numel(),
            )
            if feat is not None:
                feats.append(feat)
    streamed = torch.cat(feats, dim=1)
    assert streamed.shape == offline.shape
    torch.testing.assert_close(streamed, offline, rtol=0, atol=1e-5)


@pytest.mark.execution_timeout(20)
@pytest.mark.parametrize("chunk", [1, 32, 128, 512])
def test_Speech2TextStreaming_chunked(st_config_file_streaming, chunk):
    speech2text = Speech2TextStreaming(
        st_train_config=st_config_file_streaming, beam_size=1
    )
    speech = np.random.default_rng(0).standard_normal(10000)
    for start in range(0, len(speech), chunk):
        results = speech2text(
            speech[start : start + chunk], is_final=start + chunk >= len(speech)
        )
    assert results
    for text, token, token_int, hyp in results:
        assert isinstance(text, str)
        assert isinstance(token[0], str)
        assert isinstance(token_int[0], int)
        assert isinstance(hyp, Hypothesis)
