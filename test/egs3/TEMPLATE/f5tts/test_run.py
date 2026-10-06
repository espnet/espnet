"""Tests for the egs3/TEMPLATE/f5tts scaffold configs and runner wiring."""

from argparse import Namespace
from pathlib import Path

import pytest
from hydra.utils import get_method

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
    for key in ("dataset", "model", "optimizer", "scheduler", "dataloader"):
        assert key in config
        assert config[key] is None
    assert config.trainer.accelerator == "auto"
    assert config.parallel.n_workers == 1


def test_inference_scaffold_names_the_runner_and_leaves_the_model_open() -> None:
    config = load_default_config("inference.yaml", PACKAGE)

    assert config.model is None
    assert config.input_key is None
    assert config.output_fn is None
    assert (
        config.provider._target_
        == "espnet3.systems.base.inference_provider.InferenceProvider"
    )
    assert (
        config.runner._target_
        == "espnet3.systems.base.inference_runner.InferenceRunner"
    )


def test_template_output_fn_is_importable() -> None:
    """The helper a recipe copies into its ``src/`` is one this template ships."""
    build_output = get_method("egs3.TEMPLATE.f5tts.src.inference.build_output")
    assert callable(build_output)


def test_metrics_scaffold_enables_no_metric() -> None:
    config = load_default_config("metrics.yaml", PACKAGE)
    assert not config.metrics


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


def test_main_runs_the_data_stages(recipe_dir) -> None:
    """``run.py`` dispatches the two F5-TTS data stages on a real recipe."""
    import numpy as np
    import soundfile as sf

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

    main(
        args=_run_arguments(
            # Given out of order on purpose: execution follows DEFAULT_STAGES.
            stages=["create_token_list", "remove_long_short"],
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
