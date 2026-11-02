import string
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pytest
import soundfile
import torch

from espnet2.bin.cls_inference import Classification, get_parser, inference, main
from espnet2.tasks.cls import CLSTask


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


@pytest.fixture()
def token_list(tmp_path: Path):
    with (tmp_path / "tokens.txt").open("w") as f:
        for c in string.ascii_letters:
            f.write(f"{c}\n")
        f.write("<unk>\n")
    return tmp_path / "tokens.txt"


@pytest.fixture()
def multi_class_config_file(tmp_path: Path, token_list):
    # Write default configuration file
    CLSTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "multi_class"),
            "--token_list",
            str(token_list),
            "--decoder",
            "linear",
        ]
    )
    return tmp_path / "multi_class" / "config.yaml"


def test_Classification_from_pretrained_tag(monkeypatch, multi_class_config_file):
    class FakeDownloader:
        def download_and_unpack(self, model_tag):
            assert model_tag == "espnet/some_classifier"
            return {"classification_train_config": str(multi_class_config_file)}

    monkeypatch.setattr("espnet_model_zoo.downloader.ModelDownloader", FakeDownloader)
    classification = Classification.from_pretrained(
        "espnet/some_classifier", batch_size=1
    )
    pred, score, pred_str = classification(np.random.randn(1000))
    assert score.shape == (52,), score.shape


@pytest.fixture()
def multi_label_config_file(tmp_path: Path, token_list):
    # Write default configuration file
    CLSTask.main(
        cmd=[
            "--dry_run",
            "true",
            "--output_dir",
            str(tmp_path / "multi_label"),
            "--token_list",
            str(token_list),
            "--decoder",
            "linear",
        ]
    )
    return tmp_path / "multi_label" / "config.yaml"


@pytest.mark.execution_timeout(20)
def test_Classification_multiclass(multi_class_config_file):
    classification = Classification(
        classification_train_config=multi_class_config_file, batch_size=1
    )
    speech = np.random.randn(1000)
    pred, score, pred_str = classification(speech)
    assert pred[0] < 52, pred[0]
    assert score.shape == (52,), score.shape
    assert torch.all(score <= 1), score
    assert torch.all(score >= 0), score
    assert torch.allclose(torch.sum(score), torch.tensor(1.0)), score


@pytest.mark.execution_timeout(20)
def test_Classification_multilabel(multi_label_config_file):
    classification = Classification(
        classification_train_config=multi_label_config_file, batch_size=1
    )
    speech = np.random.randn(1000)
    pred, score, pred_str = classification(speech)
    assert all([p < 52 for p in pred]), pred
    assert score.shape == (52,), score.shape
    assert torch.all(score <= 1), score
    assert torch.all(score >= 0), score


def test_Classification_from_pretrained(multi_class_config_file):
    # model_tag=None skips the model-zoo download and builds from the kwargs,
    # the same path every other from_pretrained in espnet2/bin takes.
    classification = Classification.from_pretrained(
        model_tag=None,
        classification_train_config=multi_class_config_file,
        batch_size=1,
    )
    pred, score, pred_str = classification(np.random.randn(1000))
    assert score.shape == (52,), score.shape


@pytest.fixture()
def data_setup(tmp_path):
    """Creates temporary files for inference and returns their paths."""
    wav_file_path = tmp_path / "utt1.wav"
    scp_file_path = tmp_path / "data.scp"

    # Create wav file
    wav = np.random.randn(16000)
    soundfile.write(wav_file_path, wav, 16000)

    # Create scp file
    with scp_file_path.open("w") as f:
        f.write(f"utt1 {wav_file_path}\n")

    output_dir_path = tmp_path / "output_dir"
    output_dir_path.mkdir(exist_ok=True)  # Create output directory

    return output_dir_path, scp_file_path


def get_args(output_dir_path, scp_file_path):
    parser = get_parser()
    defaults = {
        action.dest: action.default
        for action in parser._actions
        if action.dest != "help"
    }
    defaults.pop("config", None)
    defaults.update(
        {
            "output_dir": output_dir_path,
            "data_path_and_name_and_type": [(scp_file_path, "speech", "sound")],
        }
    )
    return defaults


@pytest.mark.execution_timeout(20)
def test_inference_multilabel(data_setup, multi_label_config_file):
    output_dir_path, scp_file_path = data_setup
    args = get_args(str(output_dir_path), str(scp_file_path))
    args.update({"classification_train_config": str(multi_label_config_file)})
    inference(**args)


@pytest.mark.execution_timeout(20)
def test_inference_multiclass(data_setup, multi_class_config_file):
    output_dir_path, scp_file_path = data_setup
    args = get_args(str(output_dir_path), str(scp_file_path))
    args.update({"classification_train_config": str(multi_class_config_file)})
    inference(**args)
