"""Tests for the espnet3 speaker system."""

from unittest.mock import patch

import pytest
from omegaconf import OmegaConf

from espnet3.systems.spk.system import SPKSystem


class TestSPKSystem:
    def test_init_with_configs(self, tmp_path):
        training_config = OmegaConf.create(
            {"exp_dir": str(tmp_path / "exp"), "stats_dir": str(tmp_path / "stats")}
        )
        system = SPKSystem(training_config=training_config)
        assert system.training_config == training_config
        assert system.exp_dir == tmp_path / "exp"

    def test_init_no_config(self):
        system = SPKSystem()
        assert system.training_config is None
        assert system.exp_dir is None

    def test_create_dataset_no_config_raises(self, tmp_path):
        training_config = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
        system = SPKSystem(training_config=training_config)
        with pytest.raises(RuntimeError, match="create_dataset.func"):
            system.create_dataset()

    def test_create_dataset_no_func_raises(self, tmp_path):
        training_config = OmegaConf.create(
            {
                "exp_dir": str(tmp_path / "exp"),
                "create_dataset": {"some_key": "some_value"},
            }
        )
        system = SPKSystem(training_config=training_config)
        with pytest.raises(RuntimeError, match="create_dataset.func"):
            system.create_dataset()

    def test_create_dataset_calls_function(self, tmp_path):
        training_config = OmegaConf.create(
            {
                "exp_dir": str(tmp_path / "exp"),
                "create_dataset": {
                    "func": "os.path.exists",
                    "path": str(tmp_path),
                },
            }
        )
        system = SPKSystem(training_config=training_config)
        result = system.create_dataset()
        assert result is True

    def test_create_dataset_rejects_args(self, tmp_path):
        training_config = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
        system = SPKSystem(training_config=training_config)
        with pytest.raises(TypeError):
            system.create_dataset("unexpected_arg")

    def test_train_delegates_to_base(self, tmp_path):
        training_config = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
        system = SPKSystem(training_config=training_config)
        with patch("espnet3.systems.base.system.train") as mock_train:
            mock_train.return_value = None
            system.train()
            mock_train.assert_called_once_with(training_config)

    def test_stage_rejects_kwargs(self, tmp_path):
        training_config = OmegaConf.create({"exp_dir": str(tmp_path / "exp")})
        system = SPKSystem(training_config=training_config)
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

    def test_eer_degenerate_labels_fallback(self, tmp_path):
        from espnet3.systems.spk.metrics.eer import EER

        metric = EER()
        data = {"score": ["0.8", "0.7"], "label": ["1", "1"]}
        result = metric(data, "test", tmp_path)
        assert result["EER"] == 1.0
        assert result["minDCF"] == 1.0


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

    def test_build_model_requires_input_size_without_frontend(self):
        import argparse

        from espnet3.systems.spk.task import SpeakerTask

        args = argparse.Namespace(frontend=None, input_size=None)
        with pytest.raises(
            ValueError, match="input_size must be specified when frontend is not used."
        ):
            SpeakerTask.build_model(args)

    def test_build_model_propagates_input_size_without_frontend(self, monkeypatch):
        import argparse

        from espnet3.systems.spk import task as spk_task

        class DummyEncoder:
            def __init__(self, input_size, **kwargs):
                self.input_size = input_size

            def output_size(self):
                return 16

        class DummyPooling:
            def __init__(self, input_size, **kwargs):
                self.input_size = input_size

            def output_size(self):
                return 8

        class DummyProjector:
            def __init__(self, input_size, **kwargs):
                self.input_size = input_size

            def output_size(self):
                return 4

        class DummyLoss:
            def __init__(self, nout, nclasses, **kwargs):
                self.nout = nout
                self.nclasses = nclasses

        class DummyModel:
            def __init__(
                self,
                frontend,
                specaug,
                normalize,
                encoder,
                pooling,
                projector,
                loss,
                **kwargs,
            ):
                self.frontend = frontend
                self.encoder = encoder

        monkeypatch.setattr(spk_task.encoder_choices, "get_class", lambda _: DummyEncoder)
        monkeypatch.setattr(spk_task.pooling_choices, "get_class", lambda _: DummyPooling)
        monkeypatch.setattr(
            spk_task.projector_choices, "get_class", lambda _: DummyProjector
        )
        monkeypatch.setattr(spk_task.loss_choices, "get_class", lambda _: DummyLoss)
        monkeypatch.setattr(spk_task, "ESPnetSpeakerModel", DummyModel)

        args = argparse.Namespace(
            frontend=None,
            input_size=80,
            specaug=None,
            normalize=None,
            encoder="dummy",
            encoder_conf={},
            pooling="dummy",
            pooling_conf={},
            projector="dummy",
            projector_conf={},
            loss="dummy",
            loss_conf={},
            spk_num=2,
            model_conf={},
            init=None,
        )
        model = spk_task.SpeakerTask.build_model(args)
        assert model.frontend is None
        assert model.encoder.input_size == 80
