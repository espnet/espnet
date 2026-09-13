"""Tests for exporting portable BEATs checkpoints from ESPnet3 runs."""

import pytest
import torch
import yaml

from espnet2.beats.encoder import BeatsEncoder, BeatsPretrainingPredictor
from espnet2.beats.espnet_model import BeatsPretrainModel
from espnet3.systems.ssl.checkpoint_export import (
    export_beats_checkpoint,
    resolve_best_checkpoints,
)

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                        | Description               |
# |--------------------------------------------------|---------------------------|
# | test_resolve_prefers_loss_checkpoints            | valid.loss top-K first.   |
# | test_resolve_falls_back_to_acc_checkpoints       | valid.acc when no loss.   |
# | test_resolve_raises_without_checkpoints          | last.ckpt is not used.    |
# | test_export_averages_and_loads_into_encoder      | Averaged encoder weights  |
# |                                                  | load into BeatsEncoder.   |
# | test_export_requires_training_config             | Missing config.yaml.      |

BEATS_CONFIG = {
    "encoder_layers": 1,
    "encoder_embed_dim": 32,
    "encoder_ffn_embed_dim": 64,
    "encoder_attention_heads": 2,
    "embed_dim": 32,
    "conv_pos": 16,
    "conv_pos_groups": 2,
    "decoder_embed_dim": 32,
    "decoder_attention_heads": 2,
    "decoder_layers": 1,
    "codebook_vocab_size": 16,
}


def _touch(path):
    path.write_bytes(b"")
    return path


def test_resolve_prefers_loss_checkpoints(tmp_path):
    loss = [
        _touch(tmp_path / "epoch1_step20_valid.loss.ckpt"),
        _touch(tmp_path / "epoch0_step10_valid.loss.ckpt"),
    ]
    _touch(tmp_path / "epoch1_step20_valid.acc.ckpt")
    _touch(tmp_path / "last.ckpt")

    assert resolve_best_checkpoints(tmp_path) == sorted(loss)


def test_resolve_falls_back_to_acc_checkpoints(tmp_path):
    acc = _touch(tmp_path / "epoch0_step10_valid.acc.ckpt")

    assert resolve_best_checkpoints(tmp_path) == [acc]


def test_resolve_raises_without_checkpoints(tmp_path):
    _touch(tmp_path / "last.ckpt")

    with pytest.raises(FileNotFoundError, match="top-K"):
        resolve_best_checkpoints(tmp_path)


def _build_model(seed):
    torch.manual_seed(seed)
    encoder = BeatsEncoder(input_size=1, beats_config=BEATS_CONFIG, is_pretraining=True)
    predictor = BeatsPretrainingPredictor(beats_config=BEATS_CONFIG)
    return BeatsPretrainModel(encoder=encoder, decoder=predictor)


def test_export_averages_and_loads_into_encoder(tmp_path):
    (tmp_path / "config.yaml").write_text(
        yaml.safe_dump({"encoder_conf": {"beats_config": BEATS_CONFIG}}),
        encoding="utf-8",
    )
    models = [_build_model(seed) for seed in (0, 1)]
    for epoch, model in enumerate(models):
        torch.save(
            {"state_dict": model.state_dict(), "epoch": epoch},
            tmp_path / f"epoch{epoch}_step{epoch}_valid.loss.ckpt",
        )
    output_path = tmp_path / "beats_encoder_iter0.pt"

    assert export_beats_checkpoint(tmp_path, output_path) == output_path

    exported = torch.load(output_path, map_location="cpu")
    assert exported["cfg"]["encoder_embed_dim"] == 32
    assert set(exported["model"]) == set(models[0].encoder.state_dict())
    weight = "patch_embedding.weight"
    expected = (
        models[0].encoder.state_dict()[weight] + models[1].encoder.state_dict()[weight]
    ) / 2
    torch.testing.assert_close(exported["model"][weight], expected)
    assert not (tmp_path / f"2avg.{output_path.name}").exists()

    encoder = BeatsEncoder(input_size=1, beats_ckpt_path=str(output_path))
    torch.testing.assert_close(encoder.state_dict()[weight], expected)


def test_export_requires_training_config(tmp_path):
    with pytest.raises(FileNotFoundError, match="config.yaml"):
        export_beats_checkpoint(tmp_path, tmp_path / "out.pt")
