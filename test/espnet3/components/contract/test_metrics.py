"""Tests for espnet3.components.contract.metrics and BaseMetric's contract hook."""

import json
import logging
from pathlib import Path

import pytest

from espnet3.components.contract import Field
from espnet3.components.contract.metrics import (
    MetricContractError,
    check_metric_declaration,
    check_metric_inputs,
    check_metric_output,
    read_fields_json,
)
from espnet3.components.metrics.base_metric import BaseMetric

# ---------------------------------------------------------------------------
# class definition time: BaseMetric.__init_subclass__
# ---------------------------------------------------------------------------


def test_declared_metric_passes_at_class_definition():
    class Good(BaseMetric):
        inputs = (Field("ref", "text"), Field("hyp", "text"))
        outputs = (Field("WER", "number"),)

        def __call__(self, data, test_name, output_dir):
            return {"WER": 0.0}

    assert Good.outputs[0].kind == "number"


def test_non_number_output_rejected_at_class_definition():
    with pytest.raises(TypeError, match="must have kind 'number'"):

        class Bad(BaseMetric):
            inputs = (Field("ref", "text"),)
            outputs = (Field("transcript", "text"),)

            def __call__(self, data, test_name, output_dir):
                return {}


def test_undeclared_metric_warns_once(caplog):
    with caplog.at_level(logging.WARNING):

        class Undeclared(BaseMetric):
            def __call__(self, data, test_name, output_dir):
                return {}

    assert any("does not declare inputs/outputs" in r.message for r in caplog.records)


def test_undeclared_metric_raises_under_strict_mode(monkeypatch):
    monkeypatch.setenv("ESPNET3_STRICT_CONTRACTS", "1")
    with pytest.raises(TypeError, match="does not declare"):

        class Undeclared(BaseMetric):
            def __call__(self, data, test_name, output_dir):
                return {}


def test_input_fields_defaults_to_class_inputs():
    class Declared(BaseMetric):
        inputs = (Field("ref", "text"), Field("hyp", "text"))
        outputs = (Field("WER", "number"),)

        def __call__(self, data, test_name, output_dir):
            return {}

    assert Declared().input_fields() == Declared.inputs


# ---------------------------------------------------------------------------
# check_metric_declaration (called directly, same rule as above)
# ---------------------------------------------------------------------------


def test_check_metric_declaration_rejects_non_number():
    class Bad:
        outputs = (Field("score", "text"),)

    with pytest.raises(TypeError, match="must have kind 'number'"):
        check_metric_declaration(Bad)


# ---------------------------------------------------------------------------
# read_fields_json
# ---------------------------------------------------------------------------


def test_read_fields_json_missing_returns_none(tmp_path: Path):
    assert read_fields_json(tmp_path) is None


def test_read_fields_json_reads_written_file(tmp_path: Path):
    payload = {
        "schema_version": 1,
        "idx_key": "utt_id",
        "fields": {"ref": {"kind": "text"}},
    }
    (tmp_path / "fields.json").write_text(json.dumps(payload), encoding="utf-8")
    assert read_fields_json(tmp_path) == payload


# ---------------------------------------------------------------------------
# check_metric_inputs
# ---------------------------------------------------------------------------


class _FakeMetric:
    def __init__(self, fields):
        self._fields = fields

    def input_fields(self):
        return self._fields


def _write_scp(test_dir: Path, name: str) -> None:
    test_dir.mkdir(parents=True, exist_ok=True)
    (test_dir / f"{name}.scp").write_text("utt1 value\n", encoding="utf-8")


def test_check_metric_inputs_succeeds_when_all_present(tmp_path: Path):
    test_dir = tmp_path / "test-clean"
    _write_scp(test_dir, "ref")
    _write_scp(test_dir, "hyp")
    metric = _FakeMetric((Field("ref", "text"), Field("hyp", "text")))

    data = check_metric_inputs(metric, None, tmp_path, "test-clean")

    assert data == {"ref": test_dir / "ref.scp", "hyp": test_dir / "hyp.scp"}


def test_check_metric_inputs_raises_with_explanation_when_missing(tmp_path: Path):
    test_dir = tmp_path / "test-clean"
    _write_scp(test_dir, "hyp")
    metric = _FakeMetric((Field("ref", "text"), Field("hyp", "text")))

    with pytest.raises(MetricContractError, match="input 'ref'") as excinfo:
        check_metric_inputs(metric, None, tmp_path, "test-clean")
    assert "ref.scp is missing" in str(excinfo.value)


def test_check_metric_inputs_uses_alias_map_from_config(tmp_path: Path):
    from types import SimpleNamespace

    test_dir = tmp_path / "test-clean"
    _write_scp(test_dir, "transcript")
    metric = _FakeMetric((Field("ref", "text"),))
    config = SimpleNamespace(inputs={"ref": "transcript"})

    data = check_metric_inputs(metric, config, tmp_path, "test-clean")

    assert data == {"ref": test_dir / "transcript.scp"}


def test_check_metric_inputs_rejects_kind_mismatch_per_fields_json(tmp_path: Path):
    test_dir = tmp_path / "test-clean"
    _write_scp(test_dir, "ref")
    (test_dir / "fields.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "idx_key": "utt_id",
                "fields": {"ref": {"kind": "audio"}},
            }
        ),
        encoding="utf-8",
    )
    metric = _FakeMetric((Field("ref", "text"),))

    with pytest.raises(MetricContractError, match="wants text but ref.scp holds audio"):
        check_metric_inputs(metric, None, tmp_path, "test-clean")


def test_check_metric_inputs_allows_null_kind_in_fields_json(tmp_path: Path):
    test_dir = tmp_path / "test-clean"
    _write_scp(test_dir, "ref")
    (test_dir / "fields.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "idx_key": "utt_id",
                "fields": {"ref": {"kind": None}},
            }
        ),
        encoding="utf-8",
    )
    metric = _FakeMetric((Field("ref", "audio"),))

    data = check_metric_inputs(metric, None, tmp_path, "test-clean")

    assert data == {"ref": test_dir / "ref.scp"}


def test_check_metric_inputs_skips_optional_missing_input(tmp_path: Path):
    test_dir = tmp_path / "test-clean"
    _write_scp(test_dir, "ref")
    metric = _FakeMetric((Field("ref", "text"), Field("prompt", "text", optional=True)))

    data = check_metric_inputs(metric, None, tmp_path, "test-clean")

    assert data == {"ref": test_dir / "ref.scp"}


def test_check_metric_inputs_warns_when_no_fields_json(tmp_path: Path, caplog):
    test_dir = tmp_path / "test-clean"
    _write_scp(test_dir, "ref")
    metric = _FakeMetric((Field("ref", "text"),))

    with caplog.at_level(logging.WARNING):
        check_metric_inputs(metric, None, tmp_path, "test-clean")

    assert any("no fields.json" in r.message for r in caplog.records)


def test_check_metric_inputs_no_declaration_returns_empty(tmp_path: Path):
    metric = _FakeMetric(())
    assert check_metric_inputs(metric, None, tmp_path, "test-clean") == {}


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
