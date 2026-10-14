"""Tests for the LACE TEMPLATE runner and its default configs."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

import egs3.TEMPLATE.lace.run as lace_run
from espnet3.systems.lace.system import LACESystem
from espnet3.utils.config_utils import load_and_merge_config, load_default_config

REPO_ROOT = Path(__file__).resolve().parents[4]
TEMPLATE_CONF = REPO_ROOT / "egs3" / "TEMPLATE" / "lace" / "conf"
RECIPE_CONF = REPO_ROOT / "egs3" / "libritts" / "lace" / "conf"


def _build_stage_args(stages, **configs):
    base = dict(
        training_config=None,
        inference_config=None,
        metrics_config=None,
        publication_config=None,
        demo_config=None,
        dry_run=False,
        write_requirements=False,
    )
    base.update(configs)
    return Namespace(stages=stages, **base)


# ---------------- default configs ----------------


def test_default_training_config_has_no_task_or_init():
    """LACE models are always Hydra targets; `init` would clobber loaded weights."""
    config = load_default_config("training.yaml", "egs3.TEMPLATE.lace")
    assert "task" not in config
    assert "init" not in config
    assert config.exp_dir == "./exp/training"  # ${recipe_dir}/exp/${self_name:}
    assert config.trainer.devices == 1


def test_default_inference_config_targets():
    config = load_default_config("inference.yaml", "egs3.TEMPLATE.lace")
    # Post-#6806 form: the Inference declares its inputs/outputs.
    assert config.get("input_key") is None and config.get("output_fn") is None
    assert config.model._target_ == "espnet3.systems.lace.inference.Inference"
    assert (
        config.provider._target_
        == "espnet3.systems.base.inference_provider.InferenceProvider"
    )


def test_libritts_training_config_resolves_against_the_codec_recipe():
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
    config = load_and_merge_config(
        RECIPE_CONF / "metrics.yaml",
        "metrics.yaml",
        default_package="egs3.TEMPLATE.lace",
    )
    (metric,) = config.metrics
    assert metric.metric._target_ == "espnet3.components.metrics.versa.VersaMetric"
    assert dict(metric.inputs) == {"wav": "resyn_audio", "ref": "dataset:wav_path"}


# ---------------- runner guardrails ----------------


@pytest.mark.parametrize("stage", ["pack_demo", "upload_demo"])
def test_demo_stages_without_training_config_fail_fast(stage, tmp_path, monkeypatch):
    """A clear ValueError, not OmegaConf's InterpolationKeyError on ${exp_tag}."""
    monkeypatch.chdir(tmp_path)
    args = _build_stage_args([stage], demo_config=TEMPLATE_CONF / "demo.yaml")

    with pytest.raises(ValueError, match=r"--training_config") as exc_info:
        lace_run.main(args=args, system_cls=LACESystem, stages=lace_run.DEFAULT_STAGES)

    assert stage in str(exc_info.value)
    assert "exp_tag" in str(exc_info.value)


def test_train_stage_with_recipe_config_passes_the_guardrail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ran = {}
    monkeypatch.setattr(
        lace_run,
        "run_stages",
        lambda **kwargs: ran.setdefault("stages", list(kwargs["stages_to_run"])),
    )
    args = _build_stage_args(["train"], training_config=RECIPE_CONF / "training.yaml")

    lace_run.main(args=args, system_cls=LACESystem, stages=lace_run.DEFAULT_STAGES)

    assert ran["stages"] == ["train"]


@pytest.mark.parametrize("name", ["publication.yaml", "demo.yaml"])
def test_libritts_publication_and_demo_configs_resolve(name):
    # demo.yaml interpolates ${exp_tag} from the training config, so merge
    # without resolving (run.py resolves after copying exp_tag in).
    config = load_and_merge_config(
        RECIPE_CONF / name, name, default_package="egs3.TEMPLATE.lace", resolve=False
    )
    if name == "publication.yaml":
        assert config.pack_model.files.model_file.endswith("/last.ckpt")
        assert [Path(p).name for p in config.pack_model.include] == ["src", "dataset"]
    else:
        # No recipe code in the bundle, and the UI comes from the Inference fields.
        assert config.model.trust_user_code is False
        assert [o.key for o in config.ui.outputs] == ["resyn_audio"]
