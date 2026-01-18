"""Test-only demo inference helpers for mini_an4."""

from __future__ import annotations

from espnet3.systems.base.inference_runner import AbsInferenceRunner


class DummyProvider:
    """Build a no-op model for demo UI tests."""

    @staticmethod
    def build_model(config):
        _ = config
        return _DummyModel()


class _DummyModel:
    def __call__(self, *args, **kwargs):
        _ = args, kwargs
        return "dummy output"


class DummyRunner(AbsInferenceRunner):
    """Return a deterministic output without running a real model."""

    @staticmethod
    def forward(idx, dataset=None, model=None, **kwargs):
        _ = idx, model
        dummy_inputs = kwargs.get("dummy_inputs") or {}
        if dataset is not None:
            try:
                item = dataset[0]
            except Exception:
                item = None
            if isinstance(item, dict):
                for key, value in dummy_inputs.items():
                    if item.get(key) is None:
                        item[key] = value
        return {"hyp": "dummy output"}
