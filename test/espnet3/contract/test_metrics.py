"""Metric validation without model declarations or schema registries."""

import pytest

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract.metrics import MetricContractError, check_metric_inputs


class Score(BaseMetric):
    inputs = {"ref": str, "hyp": str, "weight": (float, type(None))}
    outputs = ("score",)

    def update(self, *, ref, hyp, weight=None):
        self.value = len(ref)

    def compute(self):
        return {"score": self.value}


def test_metric_direct_calls_validate_without_a_model():
    metric = Score()
    metric.update(ref=["a"], hyp=["b"])
    assert metric.compute() == {"score": 1}
    metric.update(ref=[], hyp=[], weight=None)
    assert metric.compute() == {"score": 0}


@pytest.mark.parametrize(
    "batch",
    [
        {"ref": ["a"], "hyp": []},
        {"ref": "a", "hyp": ["b"]},
        {"ref": ["a"], "hyp": [1]},
        {"ref": ["a"], "hyp": None},
        {"ref": ["a"], "hyp": ["b"], "weight": ["bad"]},
    ],
)
def test_invalid_batches_fail_before_state_changes(batch):
    metric = Score()
    with pytest.raises(MetricContractError):
        metric.update(**batch)
    assert not hasattr(metric, "value")


@pytest.mark.parametrize(
    "bindings",
    [
        {"ref": "dataset:text"},
        {"ref": "dataset:text", "hyp": "text", "typo": "text"},
        {"ref": "dataset:", "hyp": "text"},
        {"ref": "dataset:text", "hyp": ""},
    ],
)
def test_metric_bindings_reject_missing_unknown_or_empty_sources(bindings):
    with pytest.raises(MetricContractError):
        check_metric_inputs(Score(), bindings)


def test_metric_bindings_accept_dataset_columns_and_scp_stems():
    bindings = {"ref": "dataset:text", "hyp": "text"}
    assert check_metric_inputs(Score(), bindings) == bindings


@pytest.mark.parametrize("value", [True, "1", None])
def test_metric_rejects_nonnumeric_output(value):
    metric = Score()
    metric.value = value
    with pytest.raises(MetricContractError, match="real number"):
        metric.compute()


def test_metric_rejects_missing_output():
    class Missing(Score):
        def compute(self):
            return {}

    with pytest.raises(MetricContractError, match="missing"):
        Missing().compute()


@pytest.mark.parametrize("outputs", [(), ("score", "score"), {"score": float}])
def test_metric_requires_valid_declarations(outputs):
    with pytest.raises(MetricContractError):
        type("Invalid", (Score,), {"outputs": outputs})()


def test_metric_requires_input_declaration():
    class Missing(Score):
        inputs = None

    with pytest.raises(MetricContractError):
        Missing()
