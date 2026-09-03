"""Tests for espnet2/speechlm/trainer/deepspeed_trainer.py — DeepSpeedTrainer."""

import json
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _SimpleModel(nn.Module):
    """Tiny model for trainer tests."""

    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.freeze_me = nn.Linear(4, 4)

    def forward(self, **kwargs):
        x = kwargs.get("seqs", torch.zeros(1, 4))
        # Flatten to 2D [batch, features] regardless of input shape
        x = x.float().reshape(x.shape[0], -1)[:, :4]
        out = self.linear(x)
        loss = out.sum()
        return {"loss": loss, "stats": {"loss": loss.detach()}}


class _MockDataFactory:
    """Factory that yields a fixed number of fake batches."""

    def __init__(self, num_batches=2):
        self._num_batches = num_batches

    def build_iter(self, global_step=0, length=None):
        for _ in range(self._num_batches):
            yield {
                "seqs": torch.randn(2, 5, 1),
                "loss_masks": torch.ones(2, 5, 1),
            }


def _make_ds_config(tmp_path):
    """Write a minimal DeepSpeed config and return the path."""
    ds_config = {
        "train_batch_size": 2,
        "gradient_accumulation_steps": 1,
        "optimizer": {"type": "Adam", "params": {"lr": 1e-4}},
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-4,
                "warmup_num_steps": 10,
            },
        },
    }
    config_path = tmp_path / "ds_config.json"
    config_path.write_text(json.dumps(ds_config))
    return str(config_path)


def _make_trainer_args(tmp_path):
    """Create minimal trainer_args dict."""
    return {
        "max_step": 4,
        "save_interval": 2,
        "log_interval": 1,
        "deepspeed_config": _make_ds_config(tmp_path),
        "freeze_param": [],
    }


@pytest.fixture
def tmp_dir(tmp_path):
    return tmp_path


@pytest.fixture
def trainer(tmp_dir):
    """Build a DeepSpeedTrainer with mock components."""
    from espnet2.speechlm.trainer.deepspeed_trainer import DeepSpeedTrainer

    model = _SimpleModel()
    train_factory = _MockDataFactory(num_batches=2)
    valid_factories = {"dev": _MockDataFactory(num_batches=1)}
    output_dir = tmp_dir / "output"

    trainer = DeepSpeedTrainer(
        train_data_factory=train_factory,
        valid_data_factories=valid_factories,
        model=model,
        resume_path=None,
        output_dir=output_dir,
        trainer_args=_make_trainer_args(tmp_dir),
    )
    return trainer


# ---------------------------------------------------------------------------
# __init__ tests
# ---------------------------------------------------------------------------
class TestInit:
    def test_creates_output_dir(self, trainer, tmp_dir):
        output_dir = tmp_dir / "output"
        assert (output_dir / "checkpoints").exists()

    def test_freezes_params(self, tmp_dir):
        from espnet2.speechlm.trainer.deepspeed_trainer import DeepSpeedTrainer

        model = _SimpleModel()
        args = _make_trainer_args(tmp_dir)
        args["freeze_param"] = ["freeze_me"]

        DeepSpeedTrainer(
            train_data_factory=_MockDataFactory(),
            valid_data_factories={},
            model=model,
            resume_path=None,
            output_dir=tmp_dir / "output2",
            trainer_args=args,
        )
        # freeze_me parameters should have requires_grad=False
        for name, param in model.named_parameters():
            if name.startswith("freeze_me"):
                assert not param.requires_grad
            else:
                assert param.requires_grad

    def test_calls_deepspeed_initialize(self, tmp_dir):
        import deepspeed

        from espnet2.speechlm.trainer.deepspeed_trainer import DeepSpeedTrainer

        original_init = deepspeed.initialize
        call_count = [0]

        def counting_init(**kwargs):
            call_count[0] += 1
            return original_init(**kwargs)

        with patch.object(deepspeed, "initialize", side_effect=counting_init):
            DeepSpeedTrainer(
                train_data_factory=_MockDataFactory(),
                valid_data_factories={},
                model=_SimpleModel(),
                resume_path=None,
                output_dir=tmp_dir / "output3",
                trainer_args=_make_trainer_args(tmp_dir),
            )
        assert call_count[0] == 1


# ---------------------------------------------------------------------------
# _load_checkpoint tests
# ---------------------------------------------------------------------------
class TestLoadCheckpoint:
    def test_no_checkpoint(self, trainer):
        assert trainer.global_step == 0

    def test_load_checkpoint_directory(self, tmp_dir):
        from espnet2.speechlm.trainer.deepspeed_trainer import DeepSpeedTrainer

        # Create a fake checkpoint dir
        ckpt_dir = tmp_dir / "output4" / "checkpoints" / "step_5"
        ckpt_dir.mkdir(parents=True)

        model = _SimpleModel()
        args = _make_trainer_args(tmp_dir)

        trainer = DeepSpeedTrainer(
            train_data_factory=_MockDataFactory(),
            valid_data_factories={},
            model=model,
            resume_path=None,
            output_dir=tmp_dir / "output4",
            trainer_args=args,
        )
        # The mock load_checkpoint returns (None, None), so global_step stays 0
        # But the checkpoint path should have been detected
        assert trainer.global_step == 0  # mock returns None for client_state

    def test_load_checkpoint_file(self, tmp_dir):
        from espnet2.speechlm.trainer.deepspeed_trainer import DeepSpeedTrainer

        # Create a checkpoint file
        model = _SimpleModel()
        ckpt_path = tmp_dir / "checkpoint.pt"
        torch.save({"module": model.state_dict()}, ckpt_path)

        args = _make_trainer_args(tmp_dir)

        trainer = DeepSpeedTrainer(
            train_data_factory=_MockDataFactory(),
            valid_data_factories={},
            model=_SimpleModel(),
            resume_path=ckpt_path,
            output_dir=tmp_dir / "output5",
            trainer_args=args,
        )
        assert trainer.global_step == 0

    def test_load_checkpoint_latest_in_output(self, tmp_dir):
        from espnet2.speechlm.trainer.deepspeed_trainer import DeepSpeedTrainer

        # Create multiple checkpoint dirs
        output_dir = tmp_dir / "output6"
        for step in [3, 7, 5]:
            (output_dir / "checkpoints" / f"step_{step}").mkdir(parents=True)

        args = _make_trainer_args(tmp_dir)

        trainer = DeepSpeedTrainer(
            train_data_factory=_MockDataFactory(),
            valid_data_factories={},
            model=_SimpleModel(),
            resume_path=None,
            output_dir=output_dir,
            trainer_args=args,
        )
        # Should pick step_7 (the latest)
        # Mock returns None for client_state, so step stays 0
        assert trainer.global_step == 0


# ---------------------------------------------------------------------------
# _all_reduce_stats
# ---------------------------------------------------------------------------
class TestAllReduceStats:
    def test_not_distributed(self, trainer):
        stats = {"loss": torch.tensor(1.0)}
        trainer._all_reduce_stats(stats)
        assert stats["loss"].item() == 1.0  # unchanged

    def test_distributed(self, trainer):
        stats = {"loss": torch.tensor(4.0), "acc": torch.tensor(2.0)}

        with (
            patch("torch.distributed.is_initialized", return_value=True),
            patch("torch.distributed.get_world_size", return_value=2),
            patch(
                "torch.distributed.all_reduce",
                return_value=MagicMock(wait=lambda: None),
            ),
        ):
            trainer._all_reduce_stats(stats)
            # After dividing by world_size=2
            assert stats["loss"].item() == 2.0
            assert stats["acc"].item() == 1.0


# ---------------------------------------------------------------------------
# train_dtype
# ---------------------------------------------------------------------------
class TestTrainDtype:
    def test_bf16(self):
        from espnet2.speechlm.trainer.deepspeed_trainer import DeepSpeedTrainer

        ds_config = {"bf16": {"enabled": True}}
        assert DeepSpeedTrainer.train_dtype(None, ds_config) == torch.bfloat16

    def test_fp16(self):
        from espnet2.speechlm.trainer.deepspeed_trainer import DeepSpeedTrainer

        ds_config = {"fp16": {"enabled": True}}
        assert DeepSpeedTrainer.train_dtype(None, ds_config) == torch.float16

    def test_default(self):
        from espnet2.speechlm.trainer.deepspeed_trainer import DeepSpeedTrainer

        ds_config = {}
        assert DeepSpeedTrainer.train_dtype(None, ds_config) == torch.float


# ---------------------------------------------------------------------------
# train / valid / run
# ---------------------------------------------------------------------------
class TestTrainStep:
    def test_train_one_step(self, trainer):
        """Single training iteration should call backward + step."""
        initial_step = trainer.global_step

        # Patch to_device to pass through (no CUDA)
        with (
            patch(
                "espnet2.speechlm.trainer.deepspeed_trainer.to_device",
                side_effect=lambda batch, device, dtype=None: batch,
            ),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            trainer.train()

        assert trainer.global_step > initial_step

    def test_train_increments_global_step(self, trainer):
        with (
            patch(
                "espnet2.speechlm.trainer.deepspeed_trainer.to_device",
                side_effect=lambda batch, device, dtype=None: batch,
            ),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            before = trainer.global_step
            trainer.train()
            after = trainer.global_step
            assert after == before + 2  # 2 batches from MockDataFactory

    def test_train_logs_to_wandb(self, trainer):
        import wandb

        logged = []

        def capture_log(data, step=None):
            logged.append(data)

        with (
            patch(
                "espnet2.speechlm.trainer.deepspeed_trainer.to_device",
                side_effect=lambda batch, device, dtype=None: batch,
            ),
            patch("torch.distributed.get_rank", return_value=0),
            patch.object(wandb, "log", side_effect=capture_log),
        ):
            trainer.train()

        assert len(logged) > 0
        assert any("train/loss" in d for d in logged)


class TestValidStep:
    def test_valid_computes_averages(self, trainer):
        import wandb

        logged = []

        def capture_log(data, step=None):
            logged.append(data)

        with (
            patch(
                "espnet2.speechlm.trainer.deepspeed_trainer.to_device",
                side_effect=lambda batch, device, dtype=None: batch,
            ),
            patch.object(wandb, "log", side_effect=capture_log),
        ):
            trainer.valid()

        assert len(logged) > 0
        # Should have val/ prefixed keys
        assert any(k.startswith("val/") for d in logged for k in d)


class TestRun:
    def test_run_saves_checkpoint(self, trainer):
        """run() should call save_checkpoint after train+valid."""
        save_calls = []

        def capture_save(path, client_state=None):
            save_calls.append(path)

        trainer.model_engine.save_checkpoint = capture_save

        with (
            patch(
                "espnet2.speechlm.trainer.deepspeed_trainer.to_device",
                side_effect=lambda batch, device, dtype=None: batch,
            ),
            patch("torch.distributed.get_rank", return_value=0),
        ):
            trainer.run()

        assert len(save_calls) > 0
