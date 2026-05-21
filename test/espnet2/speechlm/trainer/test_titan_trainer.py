"""Tests for espnet2/speechlm/trainer/titan_trainer.py.

CPU-only coverage of module-level helpers and TitanTrainer methods that do
not require a live CUDA device or process group. Heavy paths (the real
__init__, _save_checkpoint, _load_checkpoint, train, valid) are not
exercised here — they need distributed init and FSDP2.
"""

from unittest.mock import patch

import pytest
import torch
import torch.nn as nn

from espnet2.speechlm.trainer.titan_trainer import TitanTrainer, reinit_model


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _SimpleModel(nn.Module):
    def __init__(self, with_embed=False):
        super().__init__()
        self.linear = nn.Linear(4, 4)
        self.layernorm_weight = nn.Parameter(torch.zeros(4))
        self.some_bias = nn.Parameter(torch.ones(4))
        if with_embed:
            self.embed_tokens = nn.Embedding(10, 4)


def _make_bare_trainer(**attrs):
    """Create TitanTrainer without running __init__."""
    t = TitanTrainer.__new__(TitanTrainer)
    for k, v in attrs.items():
        setattr(t, k, v)
    return t


# ---------------------------------------------------------------------------
# reinit_model
# ---------------------------------------------------------------------------
class TestReinitModel:
    def test_skipped_when_model_init_missing(self):
        model = _SimpleModel()
        linear_weight_before = model.linear.weight.detach().clone()
        reinit_model(model, {})
        assert torch.equal(model.linear.weight, linear_weight_before)

    def test_skipped_when_model_init_not_normal(self):
        model = _SimpleModel()
        linear_weight_before = model.linear.weight.detach().clone()
        reinit_model(model, {"model_init": "xavier"})
        assert torch.equal(model.linear.weight, linear_weight_before)

    def test_applies_when_normal_without_dist(self):
        model = _SimpleModel()
        with patch(
            "espnet2.speechlm.trainer.titan_trainer.dist.is_initialized",
            return_value=False,
        ):
            reinit_model(model, {"model_init": "normal", "model_init_std": 0.01})

        # Biases should be all zeros
        assert torch.allclose(model.linear.bias, torch.zeros_like(model.linear.bias))
        assert torch.allclose(model.some_bias, torch.zeros_like(model.some_bias))

        # "norm"-containing parameter should be all ones
        assert torch.allclose(
            model.layernorm_weight, torch.ones_like(model.layernorm_weight)
        )

        # Other weights drawn from N(0, std) — std tight enough to bound range
        # Use a generous bound to avoid flakiness.
        assert model.linear.weight.abs().max().item() < 1.0

    def test_defaults_std_when_not_provided(self):
        model = _SimpleModel()
        with patch(
            "espnet2.speechlm.trainer.titan_trainer.dist.is_initialized",
            return_value=False,
        ):
            # No model_init_std → default 0.02; just verify it runs and weights
            # end up finite.
            reinit_model(model, {"model_init": "normal"})
        assert torch.isfinite(model.linear.weight).all()


# ---------------------------------------------------------------------------
# Class-level attributes
# ---------------------------------------------------------------------------
class TestClassAttributes:
    def test_count_normalized_keys(self):
        # Sanity — keeping this set stable matters for correct all-reduce math.
        expected = {
            "loss",
            "ce_loss",
            "z_loss",
            "z_loss_s0",
            "z_loss_mm",
            "load_balance_loss",
            "acc_layer0",
        }
        assert set(TitanTrainer.count_normalized_keys) == expected


# ---------------------------------------------------------------------------
# _all_reduce_stats
# ---------------------------------------------------------------------------
class TestAllReduceStats:
    def test_early_return_when_dist_not_initialized(self):
        trainer = _make_bare_trainer(
            dp_pg=None,
            dp_size=1,
            device=torch.device("cpu"),
        )
        stats = {"loss": torch.tensor(2.0), "count": torch.tensor(4.0)}
        with patch(
            "espnet2.speechlm.trainer.titan_trainer.dist.is_initialized",
            return_value=False,
        ):
            trainer._all_reduce_stats(stats, grad_accum=1)

        # No mutation happened since we short-circuited.
        assert stats["loss"].item() == pytest.approx(2.0)
        assert stats["count"].item() == pytest.approx(4.0)

    def test_normalization_when_dist_initialized(self):
        trainer = _make_bare_trainer(
            dp_pg=None,
            dp_size=2,
            device=torch.device("cpu"),
        )
        stats = {
            "loss": torch.tensor(8.0),  # count-normalized
            "acc_layer0": torch.tensor(6.0),  # count-normalized
            "other_stat": torch.tensor(4.0),  # averaged over dp_size*grad_accum
            "count": torch.tensor(4.0),
        }

        def _noop_all_reduce(t, op=None, group=None, async_op=False):
            class _Handle:
                def wait(self):
                    return None

            return _Handle()

        with (
            patch(
                "espnet2.speechlm.trainer.titan_trainer.dist.is_initialized",
                return_value=True,
            ),
            patch(
                "espnet2.speechlm.trainer.titan_trainer.dist.all_reduce",
                side_effect=_noop_all_reduce,
            ),
        ):
            trainer._all_reduce_stats(stats, grad_accum=2)

        # 'count' is popped
        assert "count" not in stats
        # count-normalized: value / count
        assert stats["loss"].item() == pytest.approx(8.0 / 4.0)
        assert stats["acc_layer0"].item() == pytest.approx(6.0 / 4.0)
        # other_stat: value / (dp_size * grad_accum) = 4 / (2*2) = 1
        assert stats["other_stat"].item() == pytest.approx(1.0)

    def test_coerces_non_tensor_values(self):
        trainer = _make_bare_trainer(
            dp_pg=None,
            dp_size=1,
            device=torch.device("cpu"),
        )
        stats = {"loss": 3.0, "count": 2.0}

        def _noop_all_reduce(t, op=None, group=None, async_op=False):
            class _Handle:
                def wait(self):
                    return None

            return _Handle()

        with (
            patch(
                "espnet2.speechlm.trainer.titan_trainer.dist.is_initialized",
                return_value=True,
            ),
            patch(
                "espnet2.speechlm.trainer.titan_trainer.dist.all_reduce",
                side_effect=_noop_all_reduce,
            ),
        ):
            trainer._all_reduce_stats(stats, grad_accum=1)

        assert isinstance(stats["loss"], torch.Tensor)
        assert stats["loss"].item() == pytest.approx(3.0 / 2.0)


# ---------------------------------------------------------------------------
# _build_optimizer_scheduler
# ---------------------------------------------------------------------------
class TestBuildOptimizerScheduler:
    def _build(self, trainer_args, model=None):
        if model is None:
            model = _SimpleModel()
        trainer = _make_bare_trainer(
            model=model,
            trainer_args=trainer_args,
            max_step=100,
        )
        # Force the non-fused path to keep AdamW happy on CPU-only CI.
        with patch(
            "espnet2.speechlm.trainer.titan_trainer.torch.cuda.is_available",
            return_value=False,
        ):
            trainer._build_optimizer_scheduler()
        return trainer

    def test_builds_adamw_with_lr(self):
        trainer = self._build(
            {
                "optimizer": {"lr": 5e-4, "weight_decay": 0.05},
                "lr_scheduler": {"warmup_steps": 10},
            }
        )
        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        # LambdaLR scales lr by lambda(0) = 0, so check initial_lr (peak lr).
        assert trainer.optimizer.param_groups[0]["initial_lr"] == pytest.approx(5e-4)
        # Two groups: decay + no-decay (no_decay ends up empty here)
        assert len(trainer.optimizer.param_groups) == 2
        assert trainer.optimizer.param_groups[0]["weight_decay"] == pytest.approx(0.05)
        assert trainer.optimizer.param_groups[1]["weight_decay"] == 0.0

    def test_no_decay_group_excludes_embed_tokens(self):
        model = _SimpleModel(with_embed=True)
        trainer = self._build(
            {
                "optimizer": {"lr": 1e-4},
                "lr_scheduler": {"warmup_steps": 1},
            },
            model=model,
        )
        no_decay_ids = {id(p) for p in trainer.optimizer.param_groups[1]["params"]}
        embed_ids = {id(p) for p in model.embed_tokens.parameters()}
        # All embed params should land in the no-decay group.
        assert embed_ids.issubset(no_decay_ids)

    def test_lambda_lr_warmup_then_cosine(self):
        trainer = self._build(
            {
                "optimizer": {"lr": 1.0},  # simplify ratio inspection
                "lr_scheduler": {
                    "warmup_steps": 10,
                    "min_lr_ratio": 0.0,
                    "decay_end_step": 30,
                },
            }
        )
        sched = trainer.lr_scheduler

        # step 0: ratio = 0 / 10 = 0
        assert sched.lr_lambdas[0](0) == pytest.approx(0.0)
        # step 5: warmup → 0.5
        assert sched.lr_lambdas[0](5) == pytest.approx(0.5)
        # step 10: end of warmup → 1.0
        assert sched.lr_lambdas[0](10) == pytest.approx(1.0)
        # step 30: end of decay → min_lr_ratio (0.0)
        assert sched.lr_lambdas[0](30) == pytest.approx(0.0, abs=1e-6)
        # step 40: after decay → held at min_lr_ratio
        assert sched.lr_lambdas[0](40) == pytest.approx(0.0)

    def test_lambda_lr_respects_min_lr_ratio(self):
        trainer = self._build(
            {
                "optimizer": {"lr": 1.0},
                "lr_scheduler": {
                    "warmup_steps": 5,
                    "min_lr_ratio": 0.1,
                    "decay_end_step": 20,
                },
            }
        )
        # After decay step, hold at min_lr_ratio
        assert trainer.lr_scheduler.lr_lambdas[0](100) == pytest.approx(0.1)
        # At the end of cosine decay: cos(pi) = -1 → 0.1 + 0.5*0.9*0 = 0.1
        assert trainer.lr_scheduler.lr_lambdas[0](20) == pytest.approx(0.1, abs=1e-6)
