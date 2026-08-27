from pathlib import Path

from espnet3.utils.config_utils import load_and_merge_config, load_default_config


def test_load_default_config_train_contains_expected_targets() -> None:
    cfg = load_default_config("training.yaml", "egs3.TEMPLATE.spk")
    assert cfg.task == "espnet3.systems.spk.task.SpeakerTask"
    assert (
        cfg.dataset._target_ == "espnet3.components.data.data_organizer.DataOrganizer"
    )
    assert (
        cfg.dataset.preprocessor._target_
        == "espnet3.systems.spk.preprocessor.SpkPreprocessor"
    )
    # `DataOrganizer` owns the per-split `train` flag, so configs must not set it.
    assert "train" not in cfg.dataset.preprocessor
    assert cfg.optimizer._target_ == "torch.optim.Adam"


def test_default_training_config_selects_checkpoints_on_eer() -> None:
    cfg = load_default_config("training.yaml", "egs3.TEMPLATE.spk")
    assert list(cfg.best_model_criterion[0]) == ["valid/eer", 3, "min"]
    assert (
        cfg.trainer.callbacks[0]._target_
        == "espnet3.systems.spk.callbacks.SpeakerVerificationScoring"
    )


def test_default_training_config_uses_plain_dataloaders() -> None:
    cfg = load_default_config("training.yaml", "egs3.TEMPLATE.spk")
    # Fixed-length crops mean no ESPnet iterator factory and no shape files.
    assert cfg.dataloader.train.iter_factory is None
    assert cfg.dataloader.valid.iter_factory is None
    assert "stats_dir" not in cfg


def test_default_augmentation_is_off_so_the_template_runs_standalone() -> None:
    cfg = load_default_config("training.yaml", "egs3.TEMPLATE.spk")
    assert cfg.dataset.preprocessor.noise_apply_prob == 0.0
    assert cfg.dataset.preprocessor.rir_apply_prob == 0.0


def test_load_default_config_infer_contains_expected_targets() -> None:
    cfg = load_default_config("inference.yaml", "egs3.TEMPLATE.spk")
    assert cfg.model._target_ == "espnet3.systems.spk.inference.ESPnet2Speech2Score"
    assert list(cfg.input_key) == ["speech", "speech2"]
    # Only `test` is declared, so no training preprocessor is instantiated.
    assert set(cfg.dataset.preprocessor) == {"test"}


def test_load_default_config_metrics_scores_eer_and_min_dcf() -> None:
    cfg = load_default_config("metrics.yaml", "egs3.TEMPLATE.spk")
    targets = [entry.metric._target_ for entry in cfg.metrics]
    assert targets == [
        "espnet3.systems.spk.metrics.eer.EER",
        "espnet3.systems.spk.metrics.min_dcf.MinDCF",
    ]
    assert all(entry.metric.ref_key == "label" for entry in cfg.metrics)
    assert all(entry.metric.hyp_key == "score" for entry in cfg.metrics)


def test_load_and_merge_config_user_overrides_template_defaults(tmp_path: Path) -> None:
    user = tmp_path / "training_user.yaml"
    user.write_text(
        """
exp_tag: user_train
model:
  spk_num: 1211
  encoder: ecapa_tdnn
dataloader:
  train:
    batch_size: 64
""".strip() + "\n",
        encoding="utf-8",
    )

    cfg = load_and_merge_config(
        user,
        "training.yaml",
        default_package="egs3.TEMPLATE.spk",
    )

    assert cfg is not None
    assert cfg.model.spk_num == 1211
    assert cfg.dataloader.train.batch_size == 64
    # Template defaults survive the merge.
    assert cfg.dataloader.valid.iter_factory is None
    assert cfg.task == "espnet3.systems.spk.task.SpeakerTask"


def test_stage_list_skips_tokenizer_and_stats() -> None:
    from egs3.TEMPLATE.spk.run import DEFAULT_STAGES

    assert "train_tokenizer" not in DEFAULT_STAGES
    assert "collect_stats" not in DEFAULT_STAGES
    assert DEFAULT_STAGES[:4] == ["create_dataset", "train", "infer", "measure"]


def test_average_checkpoint_name_matches_the_inference_config() -> None:
    """The default inference config must point at a checkpoint that is written.

    `AverageCheckpointsCallback` names its output after the monitored metric,
    so a change to `best_model_criterion` silently breaks `infer` unless the
    two configs stay in sync.
    """
    training = load_default_config("training.yaml", "egs3.TEMPLATE.spk")
    inference = load_default_config("inference.yaml", "egs3.TEMPLATE.spk")

    monitor, nbest, _mode = training.best_model_criterion[0]
    expected = f"{monitor.replace('/', '.')}.ave_{nbest}best.pth"

    assert inference.model.model_file.endswith(expected)


def test_scoring_callback_is_instantiable() -> None:
    from hydra.utils import instantiate

    from espnet3.systems.spk.callbacks import SpeakerVerificationScoring

    cfg = load_default_config("training.yaml", "egs3.TEMPLATE.spk")
    callback = instantiate(cfg.trainer.callbacks[0])

    assert isinstance(callback, SpeakerVerificationScoring)
    assert callback.p_target == 0.05
