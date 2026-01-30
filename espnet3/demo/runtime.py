"""Demo runtime helpers."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
from omegaconf import DictConfig

from espnet3.demo.resolve import (
    load_infer_config,
    resolve_extra_kwargs,
    resolve_infer_kwargs,
    resolve_output_keys,
    resolve_provider_class,
    resolve_runner_class,
)

logger = logging.getLogger(__name__)


@dataclass
class DemoRuntime:
    """Container for demo inference runtime state."""

    infer_config: DictConfig | None
    model: Any
    runner_cls: Any | None
    output_keys: Dict[str, str]
    extra_kwargs: Dict[str, Any]


class SingleItemDataset:
    """Minimal dataset that exposes a single sample."""

    def __init__(self, item: Dict[str, Any]):
        """Store a single dataset item."""
        self._item = item

    def __len__(self) -> int:
        """Return dataset length."""
        return 1

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Return the item when ``idx`` is 0."""
        if idx != 0:
            raise IndexError(idx)
        return self._item


def build_runtime(demo_cfg, demo_dir: Path) -> DemoRuntime:
    """Build demo runtime from config and demo directory."""
    infer_cfg = _load_infer_config(demo_cfg, demo_dir)
    provider_cls = resolve_provider_class(demo_cfg, infer_cfg)
    runner_cls = resolve_runner_class(demo_cfg, infer_cfg)
    if provider_cls is None:
        raise RuntimeError("inference provider is not configured for this system.")
    model = None
    if provider_cls is not None:
        if infer_cfg is None:
            raise RuntimeError("infer_config is required to build the demo model.")
        model = provider_cls.build_model(infer_cfg)
    output_keys = resolve_output_keys(demo_cfg)
    extra_kwargs = resolve_infer_kwargs(infer_cfg)
    extra_kwargs.update(resolve_extra_kwargs(demo_cfg))
    return DemoRuntime(
        infer_config=infer_cfg,
        model=model,
        runner_cls=runner_cls,
        output_keys=output_keys,
        extra_kwargs=extra_kwargs,
    )


def run_inference(
    runtime: DemoRuntime,
    *,
    ui_names: List[str],
    ui_values: List[Any],
    output_names: List[str],
) -> List[Any]:
    """Run a single inference pass and map outputs for the UI."""
    inputs = dict(zip(ui_names, ui_values))
    item = _build_dataset_item(inputs)
    extras = dict(runtime.extra_kwargs)
    dataset = SingleItemDataset(item)
    result = _run_runner(runtime, dataset, extras)
    return _map_outputs(result, output_names, runtime.output_keys)


def _run_runner(
    runtime: DemoRuntime, dataset: SingleItemDataset, extras: Dict[str, Any]
):
    if runtime.runner_cls is None:
        if callable(runtime.model):
            primary = dataset[0]
            if len(primary) != 1:
                raise ValueError(
                    "Demo runner is missing; provide runner in infer_config or "
                    "reduce inputs to a single entry."
                )
            value = list(primary.values())[0]
            return runtime.model(value, **extras)
        raise RuntimeError("Demo runner is not configured and model is not callable.")
    return runtime.runner_cls.forward(0, dataset=dataset, model=runtime.model, **extras)


def _build_dataset_item(inputs: Dict[str, Any]) -> Dict[str, Any]:
    item = {key: _normalize_input(value) for key, value in inputs.items()}
    return item


def _normalize_input(value: Any) -> Any:
    # Gradio Audio returns (sample_rate, np.ndarray); normalize to float32 waveform.
    if isinstance(value, (list, tuple)) and len(value) == 2:
        sr, audio = value
        _ = sr
        if isinstance(audio, np.ndarray):
            return audio.astype(np.float32)
    return value


def _map_outputs(result: Any, output_names: List[str], output_keys: Dict[str, str]):
    if not output_keys and output_names:
        raise ValueError("output_keys is required when UI outputs are defined.")
    if output_keys:
        missing = [name for name in output_names if name not in output_keys]
        extra = [name for name in output_keys.keys() if name not in output_names]
        if missing or extra:
            missing_str = ", ".join(missing)
            extra_str = ", ".join(extra)
            raise ValueError(
                "output_keys mismatch with UI outputs. "
                f"missing={missing_str or '<none>'} "
                f"extra={extra_str or '<none>'}"
            )
    if isinstance(result, dict):
        mapped = []
        for name in output_names:
            key = output_keys.get(name)
            if key:
                mapped.append(result.get(key))
                continue
            if name in result:
                mapped.append(result.get(name))
                continue
            mapped.append(result)
        return mapped
    if len(output_names) == 1:
        return [result]
    return [result for _ in output_names]


def _load_infer_config(demo_cfg, demo_dir: Path) -> DictConfig | None:
    infer_path = getattr(demo_cfg, "infer_config", None)
    if not infer_path:
        return None
    path = Path(infer_path)
    if not path.is_absolute():
        path = demo_dir / path
    return load_infer_config(path)
