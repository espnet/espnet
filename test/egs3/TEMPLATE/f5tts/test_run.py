"""Tests for the egs3/TEMPLATE/f5tts scaffold configs and runner wiring."""

from argparse import Namespace
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from egs3.TEMPLATE.f5tts.run import DEFAULT_STAGES, build_parser, main
from espnet3.systems.f5tts.system import F5TTSSystem
from espnet3.utils.config_utils import load_and_merge_config, load_default_config

PACKAGE = "egs3.TEMPLATE.f5tts"


def _run_arguments(**overrides):
    arguments = {
        "stages": ["all"],
        "training_config": None,
        "inference_config": None,
        "metrics_config": None,
        "publication_config": None,
        "demo_config": None,
        "dry_run": False,
        "write_requirements": False,
    }
    arguments.update(overrides)
    return Namespace(**arguments)


def test_default_stages_run_manifest_consumers_after_create_dataset() -> None:
    """Execution follows list order, whatever order ``--stages`` is given in."""
    assert DEFAULT_STAGES == [
        "create_dataset",
        "remove_long_short",
        "create_token_list",
        "collect_stats",
        "train",
        "infer",
        "measure",
        "pack_model",
        "upload_model",
        "pack_demo",
        "upload_demo",
    ]


def test_every_default_stage_is_a_system_method() -> None:
    for stage in DEFAULT_STAGES:
        assert callable(getattr(F5TTSSystem, stage))


def test_parser_accepts_the_stage_names_and_config_options() -> None:
    parser = build_parser(stages=DEFAULT_STAGES)
    args = parser.parse_args(
        [
            "--stages",
            "remove_long_short",
            "create_token_list",
            "--training_config",
            "conf/training.yaml",
            "--demo_config",
            "conf/demo.yaml",
        ]
    )

    assert args.stages == ["remove_long_short", "create_token_list"]
    assert args.training_config == Path("conf/training.yaml")
    assert args.demo_config == Path("conf/demo.yaml")
    assert args.publication_config is None


def test_training_scaffold_names_the_keys_and_sets_no_model() -> None:
    """The template only names the keys; a recipe's training config is complete."""
    config = load_default_config("training.yaml", PACKAGE)

    # No ESPnet2 task bridge: an F5-TTS recipe instantiates `model._target_`.
    assert config.task is None
    for key in ("model", "optimizer", "scheduler", "dataloader"):
        assert key in config
        assert config[key] is None
    # The dataset scaffold hands DataOrganizer its preprocessor as a config, so
    # it can pass `train=` itself; a recipe fills in the splits.
    dataset = config.dataset
    assert dataset._target_ == "espnet3.components.data.data_organizer.DataOrganizer"
    assert dataset._recursive_ is False
    for key in ("train", "valid", "test", "preprocessor"):
        assert dataset[key] is None
    assert config.trainer.accelerator == "auto"
    assert config.parallel.n_workers == 1


def test_inference_scaffold_rebuilds_the_model_from_the_trained_config() -> None:
    config = load_default_config("inference.yaml", PACKAGE)

    assert config.model._target_ == "espnet3.systems.f5tts.inference.Inference"
    model = OmegaConf.to_container(config.model, resolve=False)
    # `train` writes config.yaml beside the checkpoint; no training file is named.
    assert model["train_config"] == "${exp_dir}/config.yaml"
    assert model["checkpoint_path"] == "${exp_dir}/last.ckpt"
    assert model["use_ema"] is True
    # An Inference declares its inputs and outputs; the runner refuses these.
    assert "input_key" not in config
    assert "output_fn" not in config
    assert "output_artifacts" not in config
    assert (
        config.provider._target_
        == "espnet3.systems.base.inference_provider.InferenceProvider"
    )
    assert (
        config.runner._target_
        == "espnet3.systems.base.inference_runner.InferenceRunner"
    )


def test_metrics_scaffold_is_the_f5tts_protocol() -> None:
    """WER, speaker similarity and UTMOS through VERSA; the references from the data."""
    config = load_default_config("metrics.yaml", PACKAGE)

    assert config.dataset is None  # a recipe names its test sets
    (entry,) = config.metrics
    assert entry.metric._target_ == "espnet3.systems.f5tts.metrics.versa.VersaMetric"
    assert [s.name for s in entry.metric.score_config] == [
        "fwhisper_wer",
        "speaker",
        "pseudo_mos",
    ]
    assert OmegaConf.to_container(entry.inputs) == {
        "wav": "wav",
        "ref": "dataset:ref_wav_path",
        "text": "dataset:text",
    }


def test_load_and_merge_config_user_overrides_template_defaults(tmp_path) -> None:
    user = tmp_path / "training_small.yaml"
    user.write_text(
        "model:\n"
        "  _target_: espnet3.systems.f5tts.f5tts.F5TTS\n"
        "  hidden_size: 768\n"
        "trainer:\n"
        "  max_steps: 600000\n",
        encoding="utf-8",
    )

    config = load_and_merge_config(user, "training.yaml", default_package=PACKAGE)

    assert config.exp_tag == "training_small"
    assert config.model.hidden_size == 768
    assert config.model._target_ == "espnet3.systems.f5tts.f5tts.F5TTS"
    assert config.trainer.max_steps == 600000
    # Scaffold values the user config did not touch are kept.
    assert config.trainer.accelerator == "auto"
    assert config.parallel.env == "local"


def test_load_and_merge_config_none_path_returns_none() -> None:
    assert load_and_merge_config(None, "demo.yaml", default_package=PACKAGE) is None


@pytest.mark.parametrize(
    "stage",
    ["remove_long_short", "create_token_list", "train", "infer", "pack_model"],
)
def test_main_refuses_a_stage_without_its_config(stage) -> None:
    with pytest.raises(ValueError, match="Config not provided"):
        main(args=_run_arguments(stages=[stage]), system_cls=F5TTSSystem)


def test_main_runs_the_data_and_training_stages(recipe_dir) -> None:
    """``run.py`` runs the recipe from manifests to a trained checkpoint.

    The stages are given out of order on purpose: execution follows
    ``DEFAULT_STAGES``. ``collect_stats`` and ``train`` build the data pipeline
    the way a real run does, so the template's ``dataset`` scaffold has to
    hand ``DataOrganizer`` the preprocessor as a config (``_recursive_:
    false``): Hydra would otherwise build ``CommonPreprocessor`` first, without
    the ``train`` argument it needs.
    """
    import numpy as np
    import soundfile as sf
    import yaml

    manifest_dir = recipe_dir / "data" / "manifest"
    manifest_dir.mkdir()
    for split in ("train", "valid"):
        rows = []
        for name, seconds in (("short", 0.5), ("mid", 2.0)):
            wav_path = recipe_dir / "data" / f"{split}_{name}.wav"
            sf.write(wav_path, np.zeros(int(seconds * 24000), dtype=np.float32), 24000)
            rows.append(f"{split}_{name}\t{wav_path}\tab cab\tspk\n")
        (manifest_dir / f"{split}.tsv").write_text("".join(rows), encoding="utf-8")
    (recipe_dir / "data" / "token_list" / "tokens.txt").unlink()
    # One optimizer step is enough to prove the pipeline; the fixture's
    # checkpoint is replaced by the one this run writes.
    training_yaml = recipe_dir / "conf" / "training.yaml"
    config = yaml.safe_load(training_yaml.read_text(encoding="utf-8"))
    config["trainer"]["max_steps"] = 1
    config["scheduler"]["warmup_steps"] = 0  # the schedule spans max_steps
    training_yaml.write_text(yaml.safe_dump(config), encoding="utf-8")
    (recipe_dir / "exp" / "training" / "last.ckpt").unlink()

    main(
        args=_run_arguments(
            stages=["train", "collect_stats", "create_token_list", "remove_long_short"],
            training_config=Path("conf/training.yaml"),
        ),
        system_cls=F5TTSSystem,
    )

    filtered = (recipe_dir / "data" / "manifest_filtered" / "train.tsv").read_text()
    assert [line.split("\t")[0] for line in filtered.splitlines()] == ["train_mid"]
    tokens = (recipe_dir / "data" / "token_list" / "tokens.txt").read_text()
    assert tokens.splitlines() == [
        "<blank>",
        "<unk>",
        "a",
        "b",
        "<space>",
        "c",
        "<sos/eos>",
    ]
    exp_dir = recipe_dir / "exp" / "training"
    assert (exp_dir / "stats" / "train" / "feats_shape").is_file()
    assert (exp_dir / "stats" / "valid" / "feats_shape").is_file()
    # train wrote the config beside the checkpoint it produced.
    assert (exp_dir / "config.yaml").is_file()
    assert (exp_dir / "last.ckpt").is_file()
