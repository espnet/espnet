import pytest
import torch

from espnet2.bin.tok_asr_inference import (
    build_iterator,
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


def test_validate_asr_branch_checkpoint_accepts_branch(tmp_path):
    model_file = tmp_path / "model.pth"
    torch.save({"losses.asr.weight": torch.tensor(1.0)}, model_file)

    validate_asr_branch_checkpoint(str(model_file))


def test_validate_asr_branch_checkpoint_rejects_tokenizer_only(tmp_path):
    model_file = tmp_path / "tokenizer.pth"
    torch.save({"model": {"tokenizer.weight": torch.tensor(1.0)}}, model_file)

    with pytest.raises(RuntimeError, match="parameters for the ASR branch"):
        validate_asr_branch_checkpoint(str(model_file))
