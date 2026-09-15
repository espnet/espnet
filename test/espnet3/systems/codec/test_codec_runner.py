"""Tests for the codec TEMPLATE runner's stage/config guardrails."""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path

import pytest

import egs3.TEMPLATE.codec.run as codec_run
from espnet3.systems.codec.system import CodecSystem

REPO_ROOT = Path(__file__).resolve().parents[4]
RECIPE_CONF = REPO_ROOT / "egs3" / "libritts" / "codec" / "conf"


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


@pytest.mark.parametrize("stage", ["pack_demo", "upload_demo"])
def test_demo_stages_without_training_config_fail_fast(stage, tmp_path, monkeypatch):
    """A clear ValueError, not OmegaConf's InterpolationKeyError on ${exp_tag}."""
    monkeypatch.chdir(tmp_path)
    args = _build_stage_args([stage], demo_config=RECIPE_CONF / "demo.yaml")

    with pytest.raises(ValueError, match=r"--training_config") as exc_info:
        codec_run.main(
            args=args, system_cls=CodecSystem, stages=codec_run.DEFAULT_STAGES
        )

    assert stage in str(exc_info.value)
    assert "exp_tag" in str(exc_info.value)


def test_demo_stage_with_training_config_passes_the_guardrail(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    ran = {}
    monkeypatch.setattr(
        codec_run,
        "run_stages",
        lambda **kwargs: ran.setdefault("stages", list(kwargs["stages_to_run"])),
    )
    args = _build_stage_args(
        ["pack_demo"],
        training_config=RECIPE_CONF / "training_encodec.yaml",
        demo_config=RECIPE_CONF / "demo.yaml",
    )

    codec_run.main(args=args, system_cls=CodecSystem, stages=codec_run.DEFAULT_STAGES)

    assert ran["stages"] == ["pack_demo"]
