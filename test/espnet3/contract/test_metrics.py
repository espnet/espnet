"""Tests for espnet3.contract.metrics and BaseMetric's contract hook."""

import pytest
from omegaconf import OmegaConf

from espnet3.contract import Field
from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.contract.metrics import (
    MetricContractError,
    check_metric_declaration,
    check_metric_inputs,
    check_metric_output,
)

# ---------------------------------------------------------------------------
# instantiation time: BaseMetric.__init__
# ---------------------------------------------------------------------------


class _Good(BaseMetric):
    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("WER", "number"),)

    def update(self, **values):
        pass

    def compute(self):
        return {"WER": 0.0}


def test_declared_metric_passes_at_instantiation():
    assert _Good().outputs[0].kind == "number"


class _Bad(BaseMetric):
    inputs = (Field("ref", "text"),)
    outputs = (Field("transcript", "text"),)

    def update(self, **values):
        pass

    def compute(self):
        return {}


def test_non_number_output_rejected_at_instantiation():
    with pytest.raises(TypeError, match="must have kind 'number'"):
        _Bad()


class _Undeclared(BaseMetric):
    def update(self, **values):
        pass

    def compute(self):
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

    def update(self, **values):
        pass

    def compute(self):
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

    def update(self, **values):
        pass

    def compute(self):
        return {}


def test_instance_declaration_rejects_bad_contract():
    with pytest.raises(TypeError, match="must have kind 'number'"):
        _BrokenVersaStyle()


# ---------------------------------------------------------------------------
# check_metric_declaration (called directly, same rule as above)
# ---------------------------------------------------------------------------


def test_check_metric_declaration_rejects_non_number():
    class Bad:
        outputs = (Field("score", "text"),)

    with pytest.raises(TypeError, match="must have kind 'number'"):
        check_metric_declaration(Bad)


# ---------------------------------------------------------------------------
# check_metric_inputs
# ---------------------------------------------------------------------------


class _FakeMetric:
    def __init__(self, fields):
        self.inputs = fields


def test_check_metric_inputs_validates_bindings_without_model_declarations():
    metric = _FakeMetric((Field("ref", "text"),))
    config = OmegaConf.create({"inputs": {"ref": "text"}})

    assert check_metric_inputs(metric, config.inputs) == {"ref": "text"}


def test_check_metric_inputs_does_not_inspect_model_output_names():
    metric = _FakeMetric((Field("ref", "text"),))
    config = OmegaConf.create({"inputs": {"ref": "custom_output"}})

    assert check_metric_inputs(metric, config.inputs) == {"ref": "custom_output"}


def test_check_metric_inputs_accepts_list_style_identity_inputs():
    # list form: each entry is both the declared name and its own source;
    # "text" is also the inference config's declared output name here.
    metric = _FakeMetric((Field("text", "text"),))
    config = OmegaConf.create({"inputs": ["text"]})

    assert check_metric_inputs(metric, config.inputs) == {"text": "text"}


def test_check_metric_inputs_skips_dataset_prefixed_sources():
    metric = _FakeMetric((Field("ref", "text"),))
    config = OmegaConf.create({"inputs": {"ref": "dataset:text"}})

    check_metric_inputs(metric, config.inputs)


def test_check_metric_inputs_optional_input_may_be_omitted():
    metric = _FakeMetric((Field("ref", "text"), Field("prompt", "text", optional=True)))
    config = OmegaConf.create({"inputs": {"ref": "text"}})

    assert check_metric_inputs(metric, config.inputs) == {"ref": "text"}


def test_check_metric_inputs_no_declaration_is_noop():
    class _NoInputs:
        inputs = ()

    assert check_metric_inputs(_NoInputs(), None) is None


# ---------------------------------------------------------------------------
# check_metric_inputs: inputs: is required, keyed by the declared Field
# name - no *_key constructor renaming, no fallback.
# ---------------------------------------------------------------------------


def test_check_metric_inputs_requires_inputs_in_config():
    metric = _FakeMetric((Field("ref", "text"), Field("hyp", "text")))

    with pytest.raises(MetricContractError, match="bind them in the metrics config"):
        check_metric_inputs(metric, None)


def test_check_metric_inputs_rejects_missing_required_name():
    metric = _FakeMetric((Field("ref", "text"), Field("hyp", "text")))
    config = OmegaConf.create({"inputs": {"ref": "text"}})

    with pytest.raises(MetricContractError, match="missing \\['hyp'\\]"):
        check_metric_inputs(metric, config.inputs)


def test_check_metric_inputs_rejects_undeclared_name():
    metric = _FakeMetric((Field("ref", "text"),))
    config = OmegaConf.create({"inputs": {"ref": "text", "extra": "text"}})

    with pytest.raises(MetricContractError, match="unknown \\['extra'\\]"):
        check_metric_inputs(metric, config.inputs)


def test_check_metric_inputs_accepts_dataset_and_inference_sources():
    from espnet3.systems.esp2_asr.metrics.wer import WER

    metric = WER()
    config = OmegaConf.create({"inputs": {"ref": "dataset:text", "hyp": "text"}})

    assert check_metric_inputs(metric, config.inputs) == {
        "ref": "dataset:text",
        "hyp": "text",
    }


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

    # inputs: missing entirely raises, naming the instance's own declaration
    with pytest.raises(MetricContractError, match="bind them in the metrics config"):
        check_metric_inputs(metric, None)

    # ref is optional, so leaving it out of inputs: is enough
    config = OmegaConf.create({"inputs": {"hyp": "text"}})
    assert check_metric_inputs(metric, config.inputs) == {"hyp": "text"}
