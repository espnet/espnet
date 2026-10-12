"""Tests for espnet2/speechlm/bin/train.py.

Covers the trainer-type dispatch logic plus the two standalone helpers
(get_parser, set_seed). The full main() orchestration depends on CUDA +
dist, so we only exercise the dispatch sub-paths with heavy helpers
mocked.
"""

import argparse
import random
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import yaml


@pytest.fixture
def train_module():
    """Import train.py lazily (needs stubs from conftest to be installed)."""
    from espnet2.speechlm.bin import train

    return train


# ---------------------------------------------------------------------------
# get_parser
# ---------------------------------------------------------------------------
class TestGetParser:
    def test_returns_argument_parser(self, train_module):
        parser = train_module.get_parser()
        assert isinstance(parser, argparse.ArgumentParser)

    def test_parses_minimal_valid_args(self, train_module, tmp_path):
        parser = train_module.get_parser()
        cfg = tmp_path / "cfg.yaml"
        cfg.write_text("trainer:\n  type: titan\n")
        args = parser.parse_args(
            [
                "--train-config",
                str(cfg),
                "--train-unregistered-specifier",
                "dummy",
                "--valid-unregistered-specifier",
                "dummy",
                "--stats-dir",
                str(tmp_path / "stats"),
            ]
        )
        assert args.train_config == cfg
        # default output-dir
        assert args.output_dir == Path("exp/train")

    def test_requires_train_config(self, train_module):
        parser = train_module.get_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])


# ---------------------------------------------------------------------------
# set_seed
# ---------------------------------------------------------------------------
class TestSetSeed:
    def test_reproducible_torch(self, train_module):
        train_module.set_seed(42)
        a = torch.rand(4)
        train_module.set_seed(42)
        b = torch.rand(4)
        assert torch.equal(a, b)

    def test_reproducible_numpy(self, train_module):
        train_module.set_seed(7)
        a = np.random.rand(4)
        train_module.set_seed(7)
        b = np.random.rand(4)
        assert np.array_equal(a, b)

    def test_reproducible_random(self, train_module):
        train_module.set_seed(123)
        a = [random.random() for _ in range(4)]
        train_module.set_seed(123)
        b = [random.random() for _ in range(4)]
        assert a == b

    def test_different_seeds_diverge(self, train_module):
        train_module.set_seed(1)
        a = torch.rand(4)
        train_module.set_seed(2)
        b = torch.rand(4)
        assert not torch.equal(a, b)

    def test_cuda_seed_is_safe_on_cpu(self, train_module):
        """torch.cuda.manual_seed* is a no-op on CPU-only builds."""
        train_module.set_seed(0)  # must not raise


# ---------------------------------------------------------------------------
# main() trainer-type dispatch
# ---------------------------------------------------------------------------
class TestMainDispatch:
    """Run main() with nearly everything patched.

    The goal is to exercise the dispatch branches that select between
    DeepSpeedTrainer / TitanTrainer / TitanPPTrainer. We patch every
    heavy operation (distributed init, data iterators, model build,
    wandb, actual training) so that only the dispatch logic runs.
    """

    def _make_args(self, tmp_path, train_cfg, trainer_type):
        config_path = tmp_path / "train.yaml"
        config_path.write_text(yaml.safe_dump(train_cfg))
        return [
            "train.py",
            "--train-config",
            str(config_path),
            "--output-dir",
            str(tmp_path / "out"),
            "--train-unregistered-specifier",
            "dummy",
            "--valid-unregistered-specifier",
            "dummy",
            "--stats-dir",
            str(tmp_path / "stats"),
            "--wandb-mode",
            "disabled",
        ]

    def _patches(self, train_module):
        """Common patches that neutralize heavy side-effects in main()."""
        fake_dist = MagicMock()
        fake_dist.is_initialized.return_value = True
        fake_dist.get_rank.return_value = 0
        fake_dist.get_world_size.return_value = 1
        fake_dist.broadcast.return_value = None
        fake_dist.init_process_group.return_value = None

        fake_torch_cuda = MagicMock()
        fake_torch_cuda.set_device.return_value = None

        # torch.tensor(seed).item() path needs to work — we can't fully mock
        # `torch` without breaking other stuff, so just patch the distributed
        # + cuda attrs on the real torch module.
        return [
            patch.object(train_module.torch, "distributed", fake_dist),
            patch.object(train_module.torch.cuda, "set_device", lambda *a, **k: None),
            patch.object(
                train_module,
                "DataIteratorFactory",
                return_value=MagicMock(),
            ),
            patch.object(
                train_module,
                "_all_job_types",
                {"speechlm": MagicMock()},
            ),
            patch.object(train_module.wandb, "init", MagicMock()),
            patch.object(train_module.wandb, "finish", MagicMock()),
            patch.object(
                train_module.torch,
                "tensor",
                side_effect=lambda data, **kw: MagicMock(
                    item=lambda: data[0] if isinstance(data, list) else data
                ),
            ),
        ]

    def _run(self, train_module, argv, ctx_mgrs):
        with patch.object(sys, "argv", argv):
            for m in ctx_mgrs:
                m.start()
            try:
                train_module.main()
            finally:
                for m in ctx_mgrs:
                    m.stop()

    def test_dispatch_deepspeed(self, train_module, tmp_path):
        train_cfg = {
            "trainer": {"type": "deepspeed", "max_step": 1},
            "data_loading": {
                "batchfy_method": "shuffle",
                "batch_size": 1,
                "num_workers": 0,
            },
            "job_type": "speechlm",
            "seed": 0,
        }
        argv = self._make_args(tmp_path, train_cfg, "deepspeed")
        ctxs = self._patches(train_module)

        ds_trainer_cls = MagicMock()
        ds_trainer_cls.return_value.run.return_value = None
        ds_module = MagicMock(
            DeepSpeedTrainer=ds_trainer_cls,
            init_distributed=MagicMock(),
        )

        with patch.dict(
            sys.modules,
            {
                "deepspeed": ds_module,
                "espnet2.speechlm.trainer.deepspeed_trainer": MagicMock(
                    DeepSpeedTrainer=ds_trainer_cls,
                ),
            },
        ):
            self._run(train_module, argv, ctxs)

        ds_trainer_cls.assert_called_once()
        ds_trainer_cls.return_value.run.assert_called_once()

    def test_dispatch_titan_no_pp(self, train_module, tmp_path):
        train_cfg = {
            "trainer": {"type": "titan", "max_step": 1, "titan_config": {}},
            "data_loading": {
                "batchfy_method": "shuffle",
                "batch_size": 1,
                "num_workers": 0,
            },
            "job_type": "speechlm",
            "seed": 0,
        }
        argv = self._make_args(tmp_path, train_cfg, "titan")
        ctxs = self._patches(train_module)

        titan_cls = MagicMock()
        titan_cls.return_value.run.return_value = None

        # Replace parallel_utils with a stub that reports pp_enabled=False.
        pu_module = MagicMock()
        fake_pd = MagicMock()
        fake_pd.pp_enabled = False
        fake_dp_mesh = MagicMock()
        fake_dp_mesh.get_local_rank.return_value = 0
        fake_dp_mesh.size.return_value = 1
        fake_pd.get_mesh.return_value = fake_dp_mesh
        pu_module.init_parallel_dims.return_value = (fake_pd, 0, 0)

        with patch.dict(
            sys.modules,
            {
                "espnet2.speechlm.model.speechlm.parallel_utils": pu_module,
                "espnet2.speechlm.trainer.titan_trainer": MagicMock(
                    TitanTrainer=titan_cls,
                ),
            },
        ):
            self._run(train_module, argv, ctxs)

        titan_cls.assert_called_once()
        titan_cls.return_value.run.assert_called_once()

    def test_dispatch_titan_pp(self, train_module, tmp_path):
        train_cfg = {
            "trainer": {"type": "titan", "max_step": 1, "titan_config": {"pp": 2}},
            "data_loading": {
                "batchfy_method": "shuffle",
                "batch_size": 1,
                "num_workers": 0,
            },
            "job_type": "speechlm",
            "seed": 0,
        }
        argv = self._make_args(tmp_path, train_cfg, "titan")
        ctxs = self._patches(train_module)

        pp_trainer_cls = MagicMock()
        pp_trainer_cls.return_value.run.return_value = None

        pu_module = MagicMock()
        fake_pd = MagicMock()
        fake_pd.pp_enabled = True
        fake_dp_mesh = MagicMock()
        fake_dp_mesh.get_local_rank.return_value = 0
        fake_dp_mesh.size.return_value = 1
        fake_pd.get_mesh.return_value = fake_dp_mesh
        pu_module.init_parallel_dims.return_value = (fake_pd, 0, 0)

        with patch.dict(
            sys.modules,
            {
                "espnet2.speechlm.model.speechlm.parallel_utils": pu_module,
                "espnet2.speechlm.trainer.titan_trainer_pp": MagicMock(
                    TitanPPTrainer=pp_trainer_cls,
                ),
            },
        ):
            self._run(train_module, argv, ctxs)

        pp_trainer_cls.assert_called_once()
        pp_trainer_cls.return_value.run.assert_called_once()

    def test_dispatch_unknown_type_raises(self, train_module, tmp_path):
        train_cfg = {
            "trainer": {"type": "bogus", "max_step": 1},
            "data_loading": {
                "batchfy_method": "shuffle",
                "batch_size": 1,
                "num_workers": 0,
            },
            "job_type": "speechlm",
            "seed": 0,
        }
        argv = self._make_args(tmp_path, train_cfg, "bogus")
        ctxs = self._patches(train_module)

        with pytest.raises(ValueError, match="Unknown trainer type"):
            self._run(train_module, argv, ctxs)


# ---------------------------------------------------------------------------
# record_token_layout
# ---------------------------------------------------------------------------
INTERVALS = {"text": [(0, 5)], "discrete_audio": [(5, 9)]}

# The same config in the three YAML shapes a saved config has turned up in.
# The first is what a hand-written config looks like; the other two broke the
# earlier version of this writer, which appended its record as text.
CONFIG_FORMS = {
    "block": "job_type: speechlm\nmultimodal_io:\n  text: {}\n  discrete_audio: {}\n",
    "document end marker": (
        "job_type: speechlm\nmultimodal_io:\n  text: {}\n  discrete_audio: {}\n...\n"
    ),
    "flow-style root": (
        "{job_type: speechlm, multimodal_io: {text: {}, discrete_audio: {}}}\n"
    ),
}


class TestRecordTokenLayout:
    @pytest.mark.parametrize("form", list(CONFIG_FORMS))
    def test_the_config_still_loads_and_keeps_its_order(
        self, train_module, tmp_path, form
    ):
        """Whichever YAML shape it was saved in."""
        config_path = tmp_path / "train.yaml"
        config_path.write_text(CONFIG_FORMS[form])

        train_module.record_token_layout(config_path, INTERVALS)

        saved = yaml.safe_load(config_path.read_text())
        # the order of multimodal_io is what hands out the token ids
        assert list(saved["multimodal_io"]) == ["text", "discrete_audio"]
        assert saved["vocab_intervals"] == {
            "text": [[0, 5]],
            "discrete_audio": [[5, 9]],
        }
        assert saved["job_type"] == "speechlm"

    def test_a_record_already_there_is_left_alone(self, train_module, tmp_path):
        """A resumed run must not overwrite the layout the first run recorded."""
        config_path = tmp_path / "train.yaml"
        config_path.write_text(CONFIG_FORMS["block"])
        train_module.record_token_layout(config_path, INTERVALS)
        after_first = config_path.read_text()

        train_module.record_token_layout(config_path, {"text": [(99, 100)]})

        assert config_path.read_text() == after_first

    def test_the_record_survives_a_trip_through_the_loader(
        self, train_module, tmp_path
    ):
        """Tuples in, lists out: the comparison on load is against plain YAML."""
        config_path = tmp_path / "train.yaml"
        config_path.write_text(CONFIG_FORMS["block"])

        train_module.record_token_layout(config_path, INTERVALS)

        layout = yaml.safe_load(config_path.read_text())["vocab_intervals"]
        assert all(
            isinstance(pair, list) and all(isinstance(x, int) for x in pair)
            for pairs in layout.values()
            for pair in pairs
        )
