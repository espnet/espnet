import numpy as np
import pytest
import soundfile
import torch
import yaml
from omniio import kaldi as kaldi_io

from espnet2.bin.tok_downstream_inference import (
    build_iterator,
    inference_reconstruction,
    validate_downstream_branch_checkpoint,
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
        None,
        str(key_file),
        "float32",
        0,
        0,
    )
    keys, batch = next(iter(iterator))

    assert keys == ["utt2"]
    assert batch["speech"].tolist() == [[4, 5]]
    assert batch["speech_lengths"].tolist() == [2]


def test_build_iterator_reads_speaker_embeddings(tmp_path):
    token_file = tmp_path / "token"
    token_file.write_text("utt1 1 2 3\n", encoding="utf-8")
    spembs_scp = tmp_path / "spembs.scp"
    with kaldi_io.WriteHelper(
        f"ark,scp:{tmp_path / 'spembs.ark'},{spembs_scp}"
    ) as writer:
        writer["utt1"] = np.array([0.1, 0.2], dtype=np.float32)

    iterator = build_iterator(
        GANSpeechTokenizerTask,
        str(token_file),
        str(spembs_scp),
        None,
        "float32",
        0,
        0,
    )
    keys, batch = next(iter(iterator))

    assert keys == ["utt1"]
    assert batch["speech"].tolist() == [[1, 2, 3]]
    np.testing.assert_allclose(batch["spembs"].numpy(), [[0.1, 0.2]])


@pytest.mark.parametrize(
    ("sample_rate", "frontend_conf", "expected_sample_rate"),
    (("16k", {}, 16000), (None, {"fs": 24000}, 24000)),
)
def test_inference_reconstruction_sample_rate(
    tmp_path, sample_rate, frontend_conf, expected_sample_rate
):
    class Model:
        @staticmethod
        def synthesize_from_tokens(token_ids, token_lengths, spembs):
            return torch.zeros(1, 1, 4), torch.tensor([4])

    train_config = tmp_path / "config.yaml"
    train_config.write_text(
        yaml.safe_dump({"frontend_conf": frontend_conf}), encoding="utf-8"
    )
    iterator = [
        (
            ["utt1"],
            {
                "speech": torch.tensor([[1, 2]]),
                "speech_lengths": torch.tensor([2]),
                "spembs": torch.zeros(1, 2),
            },
        )
    ]

    inference_reconstruction(
        Model(), iterator, tmp_path, "cpu", str(train_config), sample_rate
    )

    _, actual_sample_rate = soundfile.read(tmp_path / "wav" / "utt1.wav")
    assert actual_sample_rate == expected_sample_rate


def test_inference_reconstruction_requires_sample_rate(tmp_path):
    train_config = tmp_path / "config.yaml"
    train_config.write_text(yaml.safe_dump({"frontend_conf": {}}), encoding="utf-8")

    with pytest.raises(ValueError, match="sample_rate must be provided"):
        inference_reconstruction(None, [], tmp_path, "cpu", str(train_config), None)


@pytest.mark.parametrize("key", ("/tmp/utt1", "../utt1", "dir/utt1", "", ".", ".."))
def test_inference_reconstruction_rejects_non_filename_key(tmp_path, key):
    with pytest.raises(ValueError, match="Invalid utterance key for WAV output"):
        inference_reconstruction(
            None, [([key], {})], tmp_path, "cpu", "unused.yaml", 16000
        )


@pytest.mark.parametrize(
    ("mode", "checkpoint"),
    (
        ("asr", {"losses.asr.weight": torch.tensor(1.0)}),
        (
            "reconstruction",
            {"model": {"module.losses.reconstruction.weight": torch.tensor(1.0)}},
        ),
    ),
)
def test_validate_downstream_branch_checkpoint_accepts_branch(
    tmp_path, mode, checkpoint
):
    model_file = tmp_path / "model.pth"
    torch.save(checkpoint, model_file)

    validate_downstream_branch_checkpoint(str(model_file), mode)


@pytest.mark.parametrize("mode", ("asr", "reconstruction"))
def test_validate_downstream_branch_checkpoint_rejects_tokenizer_only(tmp_path, mode):
    model_file = tmp_path / "tokenizer.pth"
    torch.save({"model": {"tokenizer.weight": torch.tensor(1.0)}}, model_file)

    with pytest.raises(
        RuntimeError,
        match=f"does not contain parameters for the {mode} downstream branch",
    ):
        validate_downstream_branch_checkpoint(str(model_file), mode)
