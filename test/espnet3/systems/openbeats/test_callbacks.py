"""Tests for the OpenBEATs Lightning callbacks."""

import types

import espnet3.systems.openbeats.callbacks as cbmod
from espnet3.systems.openbeats.callbacks import BeatsCheckpointExport

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                    | Description                  |
# |----------------------------------------------|------------------------------|
# | test_export_runs_on_rank_zero_at_train_end   | Rank 0 exports this run's    |
# |                                              | top-K, then all ranks wait.  |
# | test_export_skipped_on_other_ranks           | Other ranks only wait.       |
# | test_export_does_not_run_at_validation_end   | No export per validation.    |


def _trainer(is_global_zero, barriers):
    strategy = types.SimpleNamespace(barrier=barriers.append)
    return types.SimpleNamespace(is_global_zero=is_global_zero, strategy=strategy)


def _patch(monkeypatch, exports):
    monkeypatch.setattr(
        cbmod, "resolve_best_checkpoints", lambda trainer: ["a.ckpt", "b.ckpt"]
    )

    def fake_export(exp_dir, output_path, checkpoint_paths=None):
        exports.append((str(exp_dir), str(output_path), checkpoint_paths))

    monkeypatch.setattr(cbmod, "export_beats_checkpoint", fake_export)


def test_export_runs_on_rank_zero_at_train_end(tmp_path, monkeypatch):
    exports, barriers = [], []
    _patch(monkeypatch, exports)
    callback = BeatsCheckpointExport(tmp_path, tmp_path / "beats_encoder_iter0.pt")

    callback.on_train_end(_trainer(True, barriers), pl_module=None)

    assert exports == [
        (str(tmp_path), str(tmp_path / "beats_encoder_iter0.pt"), ["a.ckpt", "b.ckpt"])
    ]
    assert barriers == ["beats_checkpoint_export"]


def test_export_skipped_on_other_ranks(tmp_path, monkeypatch):
    exports, barriers = [], []
    _patch(monkeypatch, exports)
    callback = BeatsCheckpointExport(tmp_path, tmp_path / "out.pt")

    callback.on_train_end(_trainer(False, barriers), pl_module=None)

    assert exports == []
    assert barriers == ["beats_checkpoint_export"]


def test_export_does_not_run_at_validation_end():
    # ModelCheckpoint runs after every other callback, so at validation end
    # the top-K does not include the validation that just finished yet.
    assert "on_validation_end" not in vars(BeatsCheckpointExport)
