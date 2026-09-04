"""Tests for espnet2/speechlm/trainer/titan_trainer_pp.py.

CPU-only coverage of TitanPPTrainer methods that don't require a live
process group or pipeline schedule. Heavy paths (__init__, train,
_save_checkpoint, _load_checkpoint) need torchtitan pipeline machinery
and CUDA, which aren't available in CI.
"""

from unittest.mock import patch

import pytest
import torch

from espnet2.speechlm.trainer.titan_trainer_pp import TitanPPTrainer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
class _FakeParallelDims:
    def __init__(self, dp_replicate_enabled=False):
        self.dp_replicate_enabled = dp_replicate_enabled

    def get_mesh(self, name):
        class _Mesh:
            def __init__(self, tag):
                self.tag = tag

            def get_group(self):
                return f"group_{self.tag}"

        # name is either "fsdp" or ["dp_replicate", "fsdp"]
        return _Mesh("_".join(name) if isinstance(name, list) else name)


def _make_bare_trainer(**attrs):
    """Instantiate TitanPPTrainer without running __init__."""
    t = TitanPPTrainer.__new__(TitanPPTrainer)
    for k, v in attrs.items():
        setattr(t, k, v)
    return t


# ---------------------------------------------------------------------------
# valid — pure no-op
# ---------------------------------------------------------------------------
class TestValid:
    def test_returns_none(self):
        trainer = _make_bare_trainer()
        assert trainer.valid() is None


# ---------------------------------------------------------------------------
# _pp_checkpoint_dir
# ---------------------------------------------------------------------------
class TestPpCheckpointDir:
    @pytest.mark.parametrize(
        "pp_rank, pp_degree, step, expected",
        [
            (0, 1, 5, "checkpoints/step_5/pp_00_01"),
            (0, 4, 100, "checkpoints/step_100/pp_00_04"),
            (3, 4, 12, "checkpoints/step_12/pp_03_04"),
            (10, 16, 99999, "checkpoints/step_99999/pp_10_16"),
        ],
    )
    def test_path_format(self, tmp_path, pp_rank, pp_degree, step, expected):
        trainer = _make_bare_trainer(
            output_dir=tmp_path,
            pp_rank=pp_rank,
            pp_degree=pp_degree,
        )
        got = trainer._pp_checkpoint_dir(step)
        assert got == tmp_path / expected


# ---------------------------------------------------------------------------
# _pp_process_group
# ---------------------------------------------------------------------------
class TestPpProcessGroup:
    def test_fsdp_mesh_when_no_replicate(self):
        trainer = _make_bare_trainer(
            parallel_dims=_FakeParallelDims(dp_replicate_enabled=False),
        )
        assert trainer._pp_process_group() == "group_fsdp"

    def test_dp_replicate_fsdp_mesh_when_replicate(self):
        trainer = _make_bare_trainer(
            parallel_dims=_FakeParallelDims(dp_replicate_enabled=True),
        )
        assert trainer._pp_process_group() == "group_dp_replicate_fsdp"


# ---------------------------------------------------------------------------
# _broadcast_stats_across_pp
# ---------------------------------------------------------------------------
class TestBroadcastStatsAcrossPp:
    def test_merges_keys_with_count_normalized(self):
        trainer = _make_bare_trainer(
            pp_degree=2,
            pp_pg=None,
            device=torch.device("cpu"),
        )
        stats = {"loss": torch.tensor(1.5), "my_custom": torch.tensor(7.0)}

        with patch(
            "espnet2.speechlm.trainer.titan_trainer_pp.dist.broadcast"
        ) as mock_bcast:
            mock_bcast.return_value = None
            result = trainer._broadcast_stats_across_pp(stats)

        # result has the union of count_normalized_keys and input keys
        expected_keys = set(TitanPPTrainer.count_normalized_keys) | set(stats)
        assert set(result) == expected_keys
        # All values are Python floats
        for v in result.values():
            assert isinstance(v, float)
        # Values present in stats preserved
        assert result["loss"] == pytest.approx(1.5)
        assert result["my_custom"] == pytest.approx(7.0)
        # Keys missing from stats default to 0.0
        assert result["z_loss"] == pytest.approx(0.0)

    def test_accepts_plain_float_values(self):
        """Non-tensor values in stats should be coerced to tensors."""
        trainer = _make_bare_trainer(
            pp_degree=3,
            pp_pg=None,
            device=torch.device("cpu"),
        )
        stats = {"loss": 2.0, "extra": 4.0}

        with patch(
            "espnet2.speechlm.trainer.titan_trainer_pp.dist.broadcast",
            return_value=None,
        ):
            result = trainer._broadcast_stats_across_pp(stats)

        assert result["loss"] == pytest.approx(2.0)
        assert result["extra"] == pytest.approx(4.0)

    def test_src_is_last_pp_rank(self):
        """broadcast should be called with group_src = pp_degree - 1."""
        trainer = _make_bare_trainer(
            pp_degree=4,
            pp_pg="mock_pg",
            device=torch.device("cpu"),
        )
        stats = {"loss": torch.tensor(0.5)}

        with patch(
            "espnet2.speechlm.trainer.titan_trainer_pp.dist.broadcast"
        ) as mock_bcast:
            mock_bcast.return_value = None
            trainer._broadcast_stats_across_pp(stats)

        # group_src should be 3 (pp_degree - 1) for every call
        for call in mock_bcast.call_args_list:
            assert call.kwargs["group_src"] == 3
            assert call.kwargs["group"] == "mock_pg"
