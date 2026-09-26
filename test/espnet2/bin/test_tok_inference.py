from argparse import ArgumentParser

import pytest
import torch
import yaml

from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.bin import tok_inference
from espnet2.bin.tok_inference import build_tokenizer, get_parser, main
from espnet2.tok.output import SpeechTokenizerOutput
from espnet2.tok.quantizer.abs_quantizer import (
    AbsSpeechTokenizerQuantizer,
)
from espnet2.tok.tokenizer import SpeechTokenizer


class DummyFrontend(AbsFrontend):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(float(scale)))

    def output_size(self):
        return 1

    def forward(self, input, input_lengths):
        return input.unsqueeze(-1) * self.scale, input_lengths


class DummyQuantizer(AbsSpeechTokenizerQuantizer):
    feature_dim = 1

    def __init__(self, threshold=0.0):
        super().__init__()
        self.threshold = torch.nn.Parameter(torch.tensor(float(threshold)))

    @property
    def num_clusters(self):
        return 2

    def _output(self, features, lengths):
        token_ids = (features[..., 0] > self.threshold).long()
        assignment = torch.nn.functional.one_hot(token_ids, 2).float()
        return SpeechTokenizerOutput(
            continuous=features,
            assignment=assignment,
            token_ids=token_ids,
            lengths=lengths,
        )

    def forward(self, features, feature_lengths):
        return self._output(features, feature_lengths)

    def encode(self, features, feature_lengths):
        return self._output(features, feature_lengths)


def test_get_parser():
    assert isinstance(get_parser(), ArgumentParser)


def test_main():
    with pytest.raises(SystemExit):
        main()


@pytest.mark.parametrize("prefix", ["tokenizer.", "module.tokenizer."])
def test_build_tokenizer_loads_tokenizer_state_only(tmp_path, monkeypatch, prefix):
    monkeypatch.setattr(
        tok_inference.frontend_choices,
        "get_class",
        lambda name: DummyFrontend,
    )
    monkeypatch.setattr(
        tok_inference.quantizer_choices,
        "get_class",
        lambda name: DummyQuantizer,
    )
    config_file = tmp_path / "config.yaml"
    with config_file.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            {
                "frontend": "dummy",
                "frontend_conf": {"scale": 2.0},
                "quantizer": "dummy",
                "quantizer_conf": {"threshold": 0.5},
                "tokenizer_conf": {"freeze_epochs": 1},
            },
            stream,
        )

    expected = SpeechTokenizer(
        DummyFrontend(scale=3.0),
        DummyQuantizer(threshold=1.5),
        freeze_epochs=1,
    )
    checkpoint = {
        "model": {
            f"{prefix}{name}": value.clone()
            for name, value in expected.state_dict().items()
        }
    }
    checkpoint["model"]["objectives.unused"] = torch.tensor(10.0)
    model_file = tmp_path / "model.pth"
    torch.save(checkpoint, model_file)

    tokenizer = build_tokenizer(
        str(config_file), str(model_file), device="cpu", dtype="float32"
    )

    assert not tokenizer.training
    for name, value in expected.state_dict().items():
        torch.testing.assert_close(tokenizer.state_dict()[name], value)


def test_build_tokenizer_rejects_checkpoint_without_tokenizer_state(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(
        tok_inference.frontend_choices,
        "get_class",
        lambda name: DummyFrontend,
    )
    monkeypatch.setattr(
        tok_inference.quantizer_choices,
        "get_class",
        lambda name: DummyQuantizer,
    )
    config_file = tmp_path / "config.yaml"
    with config_file.open("w", encoding="utf-8") as stream:
        yaml.safe_dump(
            {"frontend": "dummy", "quantizer": "dummy"},
            stream,
        )
    model_file = tmp_path / "model.pth"
    torch.save({"model": {"objectives.unused": torch.tensor(1.0)}}, model_file)

    with pytest.raises(RuntimeError, match=r"No tokenizer\.\* parameters"):
        build_tokenizer(
            str(config_file), str(model_file), device="cpu", dtype="float32"
        )


class DummyInferenceTokenizer:
    def encode(self, speech, speech_lengths):
        return SpeechTokenizerOutput(
            continuous=speech.unsqueeze(-1),
            assignment=torch.empty(2, 3, 0),
            token_ids=torch.tensor([[3, 2, 1], [4, 5, 6]]),
            lengths=torch.tensor([3, 2]),
        )


def test_inference_writes_tokens_and_lengths(tmp_path, monkeypatch):
    monkeypatch.setattr(
        tok_inference,
        "build_tokenizer",
        lambda *args, **kwargs: DummyInferenceTokenizer(),
    )
    monkeypatch.setattr(
        tok_inference.GANSpeechTokenizerTask,
        "build_streaming_iterator",
        lambda **kwargs: [
            (
                ["utt1", "utt2"],
                {
                    "speech": torch.zeros(2, 3),
                    "speech_lengths": torch.tensor([3, 2]),
                },
            )
        ],
    )
    output_dir = tmp_path / "tokens"

    tok_inference.inference(
        output_dir=str(output_dir),
        data_path_and_name_and_type=[],
        key_file=None,
        train_config="unused.yaml",
        model_file="unused.pth",
        ngpu=0,
        dtype="float32",
        batch_size=2,
        num_workers=1,
        allow_variable_data_keys=False,
        log_level="INFO",
    )

    assert (output_dir / "token").read_text(encoding="utf-8") == (
        "utt1 3 2 1\nutt2 4 5\n"
    )
    assert (output_dir / "token_length").read_text(encoding="utf-8") == (
        "utt1 3\nutt2 2\n"
    )
