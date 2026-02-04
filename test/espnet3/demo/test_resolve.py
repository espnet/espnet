from __future__ import annotations

from pathlib import Path

import pytest
from omegaconf import OmegaConf

from espnet3.demo import resolve


class DummyProvider:
    pass


class DummyRunner:
    pass


def test_resolve_absolute_path(tmp_path: Path) -> None:
    rel = resolve.resolve_absolute_path("file.txt", base=tmp_path)
    assert rel == (tmp_path / "file.txt").resolve()
    with pytest.raises(ValueError, match="absolute path could not be resolved"):
        resolve.resolve_absolute_path(None, base=tmp_path)


def test_resolve_infer_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    demo_cfg_path = tmp_path / "demo.yaml"
    demo_cfg_path.write_text("system: asr\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    path = resolve.resolve_infer_path("conf/infer.yaml")
    assert path == (tmp_path / "conf" / "infer.yaml").resolve()


def test_resolve_output_keys_from_defaults() -> None:
    cfg = OmegaConf.create({"system": "asr"})
    mapping = resolve.resolve_output_keys(cfg)
    assert mapping.get("text") == "hyp"


def test_resolve_extra_kwargs_from_config() -> None:
    cfg = OmegaConf.create({"extra_kwargs": {"beam_size": 1}})
    mapping = resolve.resolve_extra_kwargs(cfg)
    assert mapping == {"beam_size": 1}


def test_resolve_infer_kwargs_from_config() -> None:
    infer_cfg = OmegaConf.create({"input_key": "speech", "output_fn": "src.infer.fn"})
    mapping = resolve.resolve_infer_kwargs(infer_cfg)
    assert mapping == {"input_key": "speech", "output_fn_path": "src.infer.fn"}


def test_resolve_provider_runner_class_from_system() -> None:
    cfg = OmegaConf.create({"system": "asr"})
    provider_cls = resolve.resolve_provider_class(cfg)
    runner_cls = resolve.resolve_runner_class(cfg)
    assert provider_cls.__name__ == "InferenceProvider"
    assert runner_cls.__name__ == "InferenceRunner"


def test_resolve_provider_runner_class_from_infer_cfg() -> None:
    demo_cfg = OmegaConf.create({"system": "dummy"})
    infer_cfg = OmegaConf.create(
        {
            "provider": {
                "_target_": "test.espnet3.demo.test_resolve.DummyProvider",
            },
            "runner": {
                "_target_": "test.espnet3.demo.test_resolve.DummyRunner",
            },
        }
    )
    provider_cls = resolve.resolve_provider_class(demo_cfg, infer_cfg)
    runner_cls = resolve.resolve_runner_class(demo_cfg, infer_cfg)
    assert provider_cls is DummyProvider
    assert runner_cls is DummyRunner
