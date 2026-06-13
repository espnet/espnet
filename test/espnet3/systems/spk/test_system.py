"""Tests for the espnet3 speaker system."""

from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from espnet3.systems.spk.system import SPKSystem


class TestSPKSystem:
    def test_init_with_configs(self, tmp_path):
        train_config = OmegaConf.create(
            {"exp_dir": str(tmp_path / "exp"), "stats_dir": str(tmp_path / "stats")}
        )
        system = SPKSystem(train_config=train_config)
        assert system.train_config == train_config
        assert system.exp_dir == tmp_path / "exp"

    def test_init_no_config(self):
        system = SPKSystem()
        assert system.train_config is None
        assert system.exp_dir is None

    def test_create_dataset_no_config_raises(self, tmp_path):
        train_config = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
        system = SPKSystem(train_config=train_config)
        with pytest.raises(RuntimeError, match="create_dataset.func"):
            system.create_dataset()

    def test_create_dataset_no_func_raises(self, tmp_path):
        train_config = OmegaConf.create(
            {
                "exp_dir": str(tmp_path / "exp"),
                "create_dataset": {"some_key": "some_value"},
            }
        )
        system = SPKSystem(train_config=train_config)
        with pytest.raises(RuntimeError, match="create_dataset.func"):
            system.create_dataset()

    def test_create_dataset_calls_function(self, tmp_path):
        train_config = OmegaConf.create(
            {
                "exp_dir": str(tmp_path / "exp"),
                "create_dataset": {
                    "func": "os.path.exists",
                    "path": str(tmp_path),
                },
            }
        )
        system = SPKSystem(train_config=train_config)
        result = system.create_dataset()
        assert result is True

    def test_create_dataset_rejects_args(self, tmp_path):
        train_config = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
        system = SPKSystem(train_config=train_config)
        with pytest.raises(TypeError):
            system.create_dataset("unexpected_arg")

    def test_train_delegates_to_base(self, tmp_path):
        train_config = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
        system = SPKSystem(train_config=train_config)
        with patch("espnet3.systems.base.system.train") as mock_train:
            mock_train.return_value = None
            system.train()
            mock_train.assert_called_once_with(train_config)

    def test_stage_rejects_kwargs(self, tmp_path):
        train_config = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
        system = SPKSystem(train_config=train_config)
        with pytest.raises(TypeError):
            system.create_dataset(unexpected=True)


class TestEERMetric:
    def test_eer_computation(self, tmp_path):
        from espnet3.systems.spk.metrics.eer import EER

        metric = EER()
        # Perfect separation: target scores high, non-target scores low
        scores = ["0.9", "0.85", "0.1", "0.05"]
        labels = ["1", "1", "0", "0"]
        data = {"score": scores, "label": labels}

        result = metric(data, "test_set", tmp_path)
        assert "EER" in result
        assert "minDCF" in result
        assert result["EER"] == 0.0
        assert (tmp_path / "test_set" / "eer_result.txt").exists()

    def test_eer_writes_result_file(self, tmp_path):
        from espnet3.systems.spk.metrics.eer import EER

        metric = EER()
        scores = ["0.6", "0.4", "0.5", "0.3"]
        labels = ["1", "0", "1", "0"]
        data = {"score": scores, "label": labels}

        metric(data, "dev", tmp_path)
        result_file = tmp_path / "dev" / "eer_result.txt"
        assert result_file.exists()
        content = result_file.read_text()
        assert "EER" in content
        assert "minDCF" in content

    def test_eer_custom_keys(self, tmp_path):
        from espnet3.systems.spk.metrics.eer import EER

        metric = EER(score_key="sim", label_key="target")
        data = {"sim": ["0.9", "0.1"], "target": ["1", "0"]}
        result = metric(data, "test", tmp_path)
        assert "EER" in result

    def test_eer_is_abs_metric(self):
        from espnet3.components.metrics.abs_metric import AbsMetric
        from espnet3.systems.spk.metrics.eer import EER

        assert issubclass(EER, AbsMetric)


class TestSpeakerTask:
    def test_import(self):
        from espnet3.systems.spk.task import SpeakerTask

        assert SpeakerTask is not None

    def test_class_choices(self):
        from espnet3.systems.spk.task import (
            SpeakerTask,
            encoder_choices,
            frontend_choices,
            loss_choices,
            pooling_choices,
        )

        assert len(SpeakerTask.class_choices_list) == 8
        assert "ecapa_tdnn" in encoder_choices.classes
        assert "rawnet3" in encoder_choices.classes
        assert "chn_attn_stat" in pooling_choices.classes
        assert "aamsoftmax" in loss_choices.classes
        assert frontend_choices.default is None  # frontend is optional

    def test_required_data_names_train(self):
        from espnet3.systems.spk.task import SpeakerTask

        names = SpeakerTask.required_data_names(train=True)
        assert "speech" in names
        assert "spk_labels" in names

    def test_required_data_names_inference(self):
        from espnet3.systems.spk.task import SpeakerTask

        names = SpeakerTask.required_data_names(train=False)
        assert "speech" in names
        assert "spk_labels" not in names

    def test_optional_data_names(self):
        from espnet3.systems.spk.task import SpeakerTask

        names = SpeakerTask.optional_data_names()
        assert "speech2" in names
        assert "trial" in names
