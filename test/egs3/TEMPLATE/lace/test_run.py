"""Tests for the LACE TEMPLATE runner and its default configs."""

from __future__ import annotations

from pathlib import Path

from espnet3.systems.lace.system import LACESystem
from espnet3.systems.launch import build_parser, default_conf_package
from espnet3.utils.config_utils import load_and_merge_config, load_default_config

REPO_ROOT = Path(__file__).resolve().parents[4]
TEMPLATE_CONF = REPO_ROOT / "egs3" / "TEMPLATE" / "lace" / "conf"
RECIPE_CONF = REPO_ROOT / "egs3" / "libritts" / "lace" / "conf"


# ---------------- default configs ----------------


def test_default_training_config_has_no_task_or_init():
    """LACE models are always Hydra targets; `init` would clobber loaded weights."""
    config = load_default_config("training.yaml", "egs3.TEMPLATE.lace")
    assert "task" not in config
    assert "init" not in config
    assert config.exp_dir == "./exp/training"  # ${recipe_dir}/exp/${self_name:}
    assert config.trainer.devices == 1


def test_default_inference_config_targets():
    """The template names the LACE ``Inference`` and the shared provider."""
    config = load_default_config("inference.yaml", "egs3.TEMPLATE.lace")
    # Post-#6806 form: the Inference declares its inputs/outputs.
    assert config.get("input_key") is None and config.get("output_fn") is None
    assert config.model._target_ == "espnet3.systems.lace.inference.Inference"
    assert (
        config.provider._target_
        == "espnet3.systems.base.inference_provider.InferenceProvider"
    )


def test_libritts_training_config_resolves():
    """The recipe training config merges over the template and targets ``LACE``."""
    config = load_and_merge_config(
        RECIPE_CONF / "training.yaml",
        "training.yaml",
        default_package="egs3.TEMPLATE.lace",
    )
    assert config.get("task") is None
    assert config.model._target_ == "espnet3.systems.lace.src.lace.LACE"
    assert (
        config.model.pretrained_model_tag
        == "NewGame/libritts_codec_train_encodec_libritts"
    )
    assert "pretrained_model_dir" not in config.model
    assert config.create_dataset.recipe_dir == "."
    assert config.data_dir == "./data"
    for split in ("train", "valid"):
        (entry,) = config.dataset[split]
        assert "data_src" not in entry  # the recipe's own dataset package


def test_libritts_inference_config_resolves():
    """The recipe inference config rebuilds from ``lace_model.yaml``."""
    config = load_and_merge_config(
        RECIPE_CONF / "inference.yaml",
        "inference.yaml",
        default_package="egs3.TEMPLATE.lace",
    )
    assert config.model._target_ == "espnet3.systems.lace.inference.Inference"
    assert config.model.train_config.endswith("/lace_model.yaml")
    assert config.get("input_key") is None and config.get("output_fn") is None
    (entry,) = config.dataset.test
    assert entry.data_src_args.inference is True


def test_libritts_metrics_config_reads_the_reference_from_the_data():
    """The metrics config binds ``ref`` to the dataset's ``wav_path``."""
    config = load_and_merge_config(
        RECIPE_CONF / "metrics.yaml",
        "metrics.yaml",
        default_package="egs3.TEMPLATE.lace",
    )
    (metric,) = config.metrics
    assert metric.metric._target_ == "espnet3.components.metrics.versa.VersaMetric"
    assert dict(metric.inputs) == {"hyp": "resyn_audio", "ref": "dataset:wav_path"}
    assert "utmos" in metric.metric.outputs


# ---------------- launch() ----------------


def test_launch_parser_exposes_the_lace_roles_and_stages():
    """``launch`` sees the LACE stages and the template conf package."""
    parser = build_parser(LACESystem)
    args = parser.parse_args(["--stages", "train", "--training_config", "x.yaml"])
    assert args.stages == ["train"]
    assert str(args.training_config) == "x.yaml"
    assert default_conf_package(LACESystem) == "egs3.TEMPLATE.lace"
    assert [s.name for s in LACESystem.stages][:5] == [
        "create_dataset",
        "collect_stats",
        "train",
        "infer",
        "measure",
    ]
