from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from espnet2.bin import tok_inference_asr
from espnet2.bin.tok_inference_asr import (
    build_iterator,
    inference,
    validate_asr_branch_checkpoint,
)
from espnet2.tasks.gan_tok import GANSpeechTokenizerTask


def test_build_iterator_reads_tokens_and_selects_keys(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("utt1 1 2 3\nutt2 4 5\n", encoding="utf-8")
    key_file = tmp_path / "keys.scp"
    key_file.write_text("utt2 anything\n", encoding="utf-8")

    iterator = build_iterator(
        GANSpeechTokenizerTask,
        str(token_file),
        str(key_file),
        "float32",
        0,
        0,
    )
    keys, batch = next(iter(iterator))

    assert keys == ["utt2"]
    assert batch["speech"].tolist() == [[4, 5]]
    assert batch["speech_lengths"].tolist() == [2]


def test_inference_writes_asr_hypothesis(tmp_path, monkeypatch):
    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            asr_model = SimpleNamespace(
                token_list=["<blank>", "a", "b", "<sos/eos>"],
                ctc_weight=0.3,
            )
            self.asr_loss = SimpleNamespace(asr_model=asr_model)

        @staticmethod
        def encode_asr_from_tokens(token_ids, token_lengths):
            return token_ids.float().unsqueeze(-1), token_lengths

    class BeamSearch:
        def to(self, **kwargs):
            return self

        def eval(self):
            return self

        def __call__(self, **kwargs):
            return [SimpleNamespace(yseq=torch.tensor([3, 1, 2, 3]))]

    class TextTokenizer:
        @staticmethod
        def tokens2text(tokens):
            return "".join(tokens)

    monkeypatch.setattr(
        GANSpeechTokenizerTask,
        "build_model_from_file",
        staticmethod(
            lambda train_config, model_file, device: (
                Model(),
                Namespace(token_type="char", bpemodel=None),
            )
        ),
    )
    monkeypatch.setattr(
        tok_inference_asr, "build_beam_search", lambda *args, **kwargs: BeamSearch()
    )
    monkeypatch.setattr(
        tok_inference_asr, "build_tokenizer", lambda **kwargs: TextTokenizer()
    )
    token_file = tmp_path / "tokens"
    token_file.write_text("utt1 1 2 3\n", encoding="utf-8")
    model_file = tmp_path / "model.pth"
    torch.save({"losses.asr.weight": torch.tensor(1.0)}, model_file)
    output = tmp_path / "output"

    inference(
        task="gan_tok",
        token_file=str(token_file),
        output_dir=str(output),
        train_config="unused.yaml",
        model_file=str(model_file),
        key_file=None,
        ngpu=0,
        dtype="float32",
        beam_size=2,
        lm_weight=0.0,
        penalty=0.0,
        maxlenratio=0.0,
        minlenratio=0.0,
        num_workers=0,
    )

    assert (output / "text").read_text(encoding="utf-8") == "utt1 ab\n"
    assert (output / "token").read_text(encoding="utf-8") == "utt1 a b\n"
    assert (output / "token_int").read_text(encoding="utf-8") == "utt1 1 2\n"


def test_validate_asr_branch_checkpoint_accepts_branch(tmp_path):
    model_file = tmp_path / "model.pth"
    torch.save({"losses.asr.weight": torch.tensor(1.0)}, model_file)

    validate_asr_branch_checkpoint(str(model_file))


def test_validate_asr_branch_checkpoint_rejects_tokenizer_only(tmp_path):
    model_file = tmp_path / "tokenizer.pth"
    torch.save({"model": {"tokenizer.weight": torch.tensor(1.0)}}, model_file)

    with pytest.raises(RuntimeError, match="parameters for the ASR branch"):
        validate_asr_branch_checkpoint(str(model_file))
