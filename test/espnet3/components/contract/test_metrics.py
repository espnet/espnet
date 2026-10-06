"""Tests for espnet3.components.contract.metrics and BaseMetric's contract hook."""

import pytest
from omegaconf import OmegaConf

from espnet3.api.inference import Field
from espnet3.components.contract.metrics import (
    MetricContractError,
    check_metric_declaration,
    check_metric_inputs,
    check_metric_output,
    declared_outputs,
)
from espnet3.components.metrics.base_metric import BaseMetric

# ---------------------------------------------------------------------------
# instantiation time: BaseMetric.__init__
# ---------------------------------------------------------------------------


class _Good(BaseMetric):
    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("WER", "number"),)

    def __call__(self, data, test_name, output_dir):
        return {"WER": 0.0}


def test_declared_metric_passes_at_instantiation():
    assert _Good().outputs[0].kind == "number"


class _Bad(BaseMetric):
    inputs = (Field("ref", "text"),)
    outputs = (Field("transcript", "text"),)

    def __call__(self, data, test_name, output_dir):
        return {}


def test_non_number_output_rejected_at_instantiation():
    with pytest.raises(TypeError, match="must have kind 'number'"):
        _Bad()


class _Undeclared(BaseMetric):
    def __call__(self, data, test_name, output_dir):
        return {}


def test_undeclared_metric_raises_at_instantiation():
    # the class itself is defined without error; only instantiating it checks
    with pytest.raises(TypeError, match="does not declare"):
        _Undeclared()


class _VersaStyle(BaseMetric):
    """A contract built in __init__ instead of declared on the class."""

    def __init__(self, score_config):
        self.score_config = score_config
        self.inputs = (
            Field("hyp", "text"),
            Field("ref", "text", optional=True),
        )
        self.outputs = tuple(Field(name, "number") for name in score_config)
        super().__init__()

    def __call__(self, data, test_name, output_dir):
        return {name: 0.0 for name in self.score_config}


def test_instance_declaration_is_checked():
    metric = _VersaStyle(["mcd", "f0"])
    assert [f.name for f in metric.outputs] == ["mcd", "f0"]
    assert metric.inputs[1].optional is True


class _BrokenVersaStyle(BaseMetric):
    """A __init__-built contract with a non-number output, for the rejection test."""

    def __init__(self):
        self.inputs = (Field("hyp", "text"),)
        self.outputs = (Field("transcript", "text"),)
        super().__init__()

    def __call__(self, data, test_name, output_dir):
        return {}


def test_instance_declaration_rejects_bad_contract():
    with pytest.raises(TypeError, match="must have kind 'number'"):
        _BrokenVersaStyle()


def test_input_sources_defaults_to_identity():
    assert _Good().input_sources() == {"ref": "ref", "hyp": "hyp"}


# ---------------------------------------------------------------------------
# check_metric_declaration (called directly, same rule as above)
# ---------------------------------------------------------------------------


def test_check_metric_declaration_rejects_non_number():
    class Bad:
        outputs = (Field("score", "text"),)

    with pytest.raises(TypeError, match="must have kind 'number'"):
        check_metric_declaration(Bad)


# ---------------------------------------------------------------------------
# declared_outputs
# ---------------------------------------------------------------------------


def test_declared_outputs_reads_the_configured_model():
    cfg = OmegaConf.create(
        {"model": {"_target_": "espnet3.systems.esp2_asr.inference.Inference"}}
    )
    assert [f.name for f in declared_outputs(cfg)] == ["text"]


def test_declared_outputs_none_for_non_inference_model():
    cfg = OmegaConf.create({"model": {"_target_": "builtins.dict"}})
    assert declared_outputs(cfg) is None


def test_declared_outputs_none_when_unset():
    assert declared_outputs(OmegaConf.create({})) is None


# ---------------------------------------------------------------------------
# check_metric_inputs
# ---------------------------------------------------------------------------


class _FakeMetric:
    def __init__(self, fields, sources=None):
        self.inputs = fields
        self._sources = sources or {f.name: f.name for f in fields}

    def input_sources(self):
        return self._sources


_INFERENCE_CFG = OmegaConf.create(
    {"model": {"_target_": "espnet3.systems.esp2_asr.inference.Inference"}}
)
_NON_INFERENCE_CFG = OmegaConf.create({"model": {"_target_": "builtins.dict"}})


def test_check_metric_inputs_passes_when_source_matches_declared_output():
    metric = _FakeMetric((Field("ref", "text"),), {"ref": "text"})
    check_metric_inputs(metric, None, _INFERENCE_CFG)


def test_check_metric_inputs_raises_when_source_is_not_a_declared_output():
    metric = _FakeMetric((Field("ref", "text"),), {"ref": "missing"})

    with pytest.raises(MetricContractError, match="wants input 'ref' -> 'missing'"):
        check_metric_inputs(metric, None, _INFERENCE_CFG)


def test_check_metric_inputs_rejects_kind_mismatch():
    metric = _FakeMetric((Field("ref", "audio"),), {"ref": "text"})

    with pytest.raises(MetricContractError, match="wants kind 'audio'"):
        check_metric_inputs(metric, None, _INFERENCE_CFG)


def test_check_metric_inputs_uses_config_inputs_override():
    metric = _FakeMetric((Field("ref", "text"),), {"ref": "missing"})
    config = OmegaConf.create({"inputs": {"ref": "text"}})

    check_metric_inputs(metric, config, _INFERENCE_CFG)


def test_check_metric_inputs_skips_dataset_prefixed_sources():
    metric = _FakeMetric((Field("ref", "text"),), {"ref": "dataset:text"})

    check_metric_inputs(metric, None, _NON_INFERENCE_CFG)


def test_check_metric_inputs_skips_optional_missing_input():
    metric = _FakeMetric(
        (Field("ref", "text"), Field("prompt", "text", optional=True)),
        {"ref": "text", "prompt": "missing"},
    )

    check_metric_inputs(metric, None, _INFERENCE_CFG)


def test_check_metric_inputs_rejects_non_inference_model():
    metric = _FakeMetric((Field("ref", "text"),), {"ref": "text"})

    with pytest.raises(MetricContractError, match="declares no outputs"):
        check_metric_inputs(metric, None, _NON_INFERENCE_CFG)


def test_check_metric_inputs_no_declaration_is_noop():
    class _NoInputs:
        inputs = ()

    check_metric_inputs(_NoInputs(), None, _NON_INFERENCE_CFG)


# ---------------------------------------------------------------------------
# check_metric_output
# ---------------------------------------------------------------------------


class _FakeOutputMetric:
    outputs = (Field("WER", "number"),)


def test_check_metric_output_accepts_number():
    check_metric_output(_FakeOutputMetric(), {"WER": 4.3})


def test_check_metric_output_rejects_non_number():
    with pytest.raises(MetricContractError, match="must be number"):
        check_metric_output(_FakeOutputMetric(), {"WER": "bad"})


def test_check_metric_output_rejects_missing_key():
    with pytest.raises(MetricContractError, match="missing"):
        check_metric_output(_FakeOutputMetric(), {})


def test_check_metric_output_no_declaration_is_noop():
    class NoOutputs:
        pass

    check_metric_output(NoOutputs(), {"anything": "goes"})


def test_check_metric_output_reads_the_instance_not_the_class():
    # score_config picks "f0" here; a class-level read would see no
    # declaration at all (VersaStyle declares nothing on the class)
    metric = _VersaStyle(["f0"])

    with pytest.raises(MetricContractError, match="missing"):
        check_metric_output(metric, {})

    check_metric_output(metric, {"f0": 4.3})


def test_check_metric_inputs_reads_the_instance_too():
    metric = _VersaStyle(["mcd"])  # inputs = (hyp (required), ref (optional))

    # default input_sources() is identity: "hyp" names no declared output
    with pytest.raises(MetricContractError, match="wants input 'hyp'"):
        check_metric_inputs(metric, None, _INFERENCE_CFG)

    # ref is optional, so a model that declares only hyp's source is enough
    metric.input_sources = lambda: {"hyp": "text", "ref": "missing"}
    check_metric_inputs(metric, None, _INFERENCE_CFG)
