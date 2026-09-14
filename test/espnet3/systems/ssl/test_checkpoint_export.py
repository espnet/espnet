"""Tests for exporting portable BEATs checkpoints from ESPnet3 runs."""

import types

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
# | test_resolve_prefers_loss_checkpoints            | valid/loss top-K first.   |
# | test_resolve_falls_back_to_acc_checkpoints       | valid/acc when no loss.   |
# | test_resolve_ignores_checkpoints_of_other_runs   | Only this run's kept      |
# |                                                  | checkpoints, even when it |
# |                                                  | kept fewer than K.        |
# | test_resolve_raises_without_kept_checkpoints     | Unmonitored/last.ckpt     |
# |                                                  | checkpoints are not used. |
# | test_export_averages_and_loads_into_encoder      | Averaged encoder weights  |
# |                                                  | load into BeatsEncoder.   |
# | test_export_requires_training_config             | Missing config.yaml.      |
# | test_export_requires_a_checkpoint_source         | No trainer, no paths.     |

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


def _trainer(kept, monitored_without_checkpoints=()):
    """Fake trainer exposing Lightning's ModelCheckpoint attributes."""
    callbacks = [
        types.SimpleNamespace(
            monitor=monitor, best_k_models={str(p): 0.0 for p in paths}
        )
        for monitor, paths in kept.items()
    ]
    callbacks += [
        types.SimpleNamespace(monitor=monitor, best_k_models={})
        for monitor in monitored_without_checkpoints
    ]
    # The last-checkpoint callback has no monitor.
    callbacks.append(types.SimpleNamespace(monitor=None, best_k_models={}))
    return types.SimpleNamespace(
        trainer=types.SimpleNamespace(checkpoint_callbacks=callbacks)
    )


def test_resolve_prefers_loss_checkpoints(tmp_path):
    loss = [
        _touch(tmp_path / "epoch1_step20_valid.loss.ckpt"),
        _touch(tmp_path / "epoch0_step10_valid.loss.ckpt"),
    ]
    acc = [_touch(tmp_path / "epoch1_step20_valid.acc.ckpt")]

    resolved = resolve_best_checkpoints(
        _trainer({"valid/acc": acc, "valid/loss": loss})
    )

    assert resolved == sorted(loss)


def test_resolve_falls_back_to_acc_checkpoints(tmp_path):
    acc = [_touch(tmp_path / "epoch0_step10_valid.acc.ckpt")]

    resolved = resolve_best_checkpoints(
        _trainer({"valid/acc": acc}, monitored_without_checkpoints=["valid/loss"])
    )

    assert resolved == acc


def test_resolve_ignores_checkpoints_of_other_runs(tmp_path):
    # A previous run left a full top-K behind; this run kept fewer than K so
    # far, so a newest-K filesystem scan would mix the two runs.
    for epoch in range(10):
        _touch(tmp_path / f"epoch{epoch}_step{epoch}0_valid.loss.ckpt")
    current = [
        _touch(tmp_path / "epoch20_step200_valid.loss.ckpt"),
        _touch(tmp_path / "epoch21_step210_valid.loss.ckpt"),
    ]

    resolved = resolve_best_checkpoints(_trainer({"valid/loss": current}))

    assert resolved == sorted(current)


def test_resolve_raises_without_kept_checkpoints(tmp_path):
    _touch(tmp_path / "epoch0_step10_valid.loss.ckpt")
    _touch(tmp_path / "last.ckpt")

    with pytest.raises(FileNotFoundError, match="kept no checkpoint"):
        resolve_best_checkpoints(_trainer({"valid/other": [tmp_path / "last.ckpt"]}))


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

    trainer = _trainer({"valid/loss": sorted(tmp_path.glob("epoch*_valid.loss.ckpt"))})

    assert export_beats_checkpoint(tmp_path, output_path, trainer) == output_path

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
        export_beats_checkpoint(tmp_path, tmp_path / "out.pt", _trainer({}))


def test_export_requires_a_checkpoint_source(tmp_path):
    (tmp_path / "config.yaml").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="trainer or checkpoint_paths"):
        export_beats_checkpoint(tmp_path, tmp_path / "out.pt")
