import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from espnet3.api.inference import Field, InferenceAPI
from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.systems.base.metric import _resolve_test_sets, measure
from espnet3.utils.scp_utils import get_class_path


class _DummyInference(InferenceAPI):
    """A model declaration covering every output name this file's metrics use."""

    inputs = (Field("speech", "audio"),)
    outputs = (
        Field("ref", "text"),
        Field("hyp", "text"),
        Field("text", "text"),
        Field("hypothesis", "text"),
    )

    def run(self, speech):
        raise NotImplementedError


_INFERENCE_CFG = OmegaConf.create(
    {"model": {"_target_": f"{__name__}._DummyInference"}}
)


class DummyMetric(BaseMetric):
    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("count", "number"),)

    def __call__(self, data, test_name, inference_dir):
        return {"count": sum(1 for _ in self.iter_inputs(data, "ref"))}


class NoKeyMetric(BaseMetric):
    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("ok", "number"),)

    def __call__(self, data, test_name, inference_dir):
        return {"ok": 1}


class PathMetric(BaseMetric):
    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("ok", "number"),)

    def __call__(self, data, test_name, inference_dir):
        task_dir = Path(inference_dir) / test_name
        assert data["ref"] == task_dir / "ref.scp"
        assert [row["hyp"] for _, row in self.iter_inputs(data, "hyp")] == ["h1"]
        return {"ok": 1}


class NotMetric:
    pass


def _write_scp(path: Path, entries):
    path.write_text("\n".join(entries), encoding="utf-8")


def _build_inputs(tmp_path: Path, **entries: list[str]) -> dict[str, Path]:
    data = {}
    for key, lines in entries.items():
        path = tmp_path / f"{key}.scp"
        path.write_text("\n".join(lines), encoding="utf-8")
        data[key] = path
    return data


def test_metric_uses_identity_inputs_and_writes_json(tmp_path):
    inference_dir = tmp_path / "infer"
    test_name = "test_a"
    task_dir = inference_dir / test_name
    task_dir.mkdir(parents=True)
    _write_scp(task_dir / "ref.scp", ["utt1 r1", "utt2 r2"])
    _write_scp(task_dir / "hyp.scp", ["utt1 h1", "utt2 h2"])

    cfg = OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "dataset": {"test": [{"name": test_name}]},
            "metrics": [
                {
                    "metric": {"_target_": f"{__name__}.DummyMetric"},
                    "inputs": {"ref": "ref", "hyp": "hyp"},
                }
            ],
        }
    )

    results = measure(cfg, _INFERENCE_CFG)

    expected_key = get_class_path(DummyMetric())
    assert results[expected_key][test_name] == {"count": 2}
    metrics_path = inference_dir / "metrics.json"
    assert metrics_path.is_file()
    assert json.loads(metrics_path.read_text(encoding="utf-8")) == results


def test_iter_inputs_rejects_length_mismatch(tmp_path):
    metric = DummyMetric()
    data = _build_inputs(
        tmp_path,
        ref=["utt1 r1", "utt2 r2"],
        hyp=["utt1 h1"],
    )

    iterator = metric.iter_inputs(data, "ref", "hyp")
    assert next(iterator) == ("utt1", {"ref": "r1", "hyp": "h1"})
    with pytest.raises(AssertionError, match="SCP length mismatch"):
        next(iterator)


def test_iter_inputs_reads_single_line_scp(tmp_path):
    metric = DummyMetric()
    data = _build_inputs(
        tmp_path,
        ref=["utt1 value1"],
    )

    assert list(metric.iter_inputs(data, "ref")) == [
        ("utt1", {"ref": "value1"}),
    ]


def test_iter_inputs_reads_multiple_aligned_lines(tmp_path):
    metric = DummyMetric()
    data = _build_inputs(
        tmp_path,
        ref=["utt1 r1", "utt2 r2"],
        hyp=["utt1 h1", "utt2 h2"],
    )

    assert list(metric.iter_inputs(data, "ref", "hyp")) == [
        ("utt1", {"ref": "r1", "hyp": "h1"}),
        ("utt2", {"ref": "r2", "hyp": "h2"}),
    ]


def test_iter_inputs_skips_blank_lines_and_accepts_empty_value(tmp_path):
    metric = DummyMetric()
    data = _build_inputs(
        tmp_path,
        ref=["utt1 value1", "", "utt2"],
    )

    assert list(metric.iter_inputs(data, "ref")) == [
        ("utt1", {"ref": "value1"}),
        ("utt2", {"ref": ""}),
    ]


def test_metric_uses_config_inputs_mapping(tmp_path):
    inference_dir = tmp_path / "infer"
    test_name = "test_a"
    task_dir = inference_dir / test_name
    task_dir.mkdir(parents=True)
    _write_scp(task_dir / "text.scp", ["utt1 r1"])
    _write_scp(task_dir / "hypothesis.scp", ["utt1 h1"])

    cfg = OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "dataset": {"test": [{"name": test_name}]},
            "metrics": [
                {
                    "metric": {"_target_": f"{__name__}.NoKeyMetric"},
                    "inputs": {"ref": "text", "hyp": "hypothesis"},
                }
            ],
        }
    )

    results = measure(cfg, _INFERENCE_CFG)

    expected_key = get_class_path(NoKeyMetric())
    assert results[expected_key][test_name] == {"ok": 1}


def test_metric_passes_lazy_scp_inputs(tmp_path):
    inference_dir = tmp_path / "infer"
    test_name = "test_a"
    task_dir = inference_dir / test_name
    task_dir.mkdir(parents=True)
    _write_scp(task_dir / "ref.scp", ["utt1 r1"])
    _write_scp(task_dir / "hyp.scp", ["utt1 h1"])

    cfg = OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "dataset": {"test": [{"name": test_name}]},
            "metrics": [
                {
                    "metric": {"_target_": f"{__name__}.PathMetric"},
                    "inputs": {"ref": "ref", "hyp": "hyp"},
                }
            ],
        }
    )

    results = measure(cfg, _INFERENCE_CFG)

    expected_key = get_class_path(PathMetric())
    assert results[expected_key][test_name] == {"ok": 1}


def test_metric_discovers_test_sets_from_inference_dir(tmp_path):
    inference_dir = tmp_path / "infer"
    for test_name in ("test_b", "test_a"):
        task_dir = inference_dir / test_name
        task_dir.mkdir(parents=True)
        _write_scp(task_dir / "ref.scp", ["utt1 r1"])
        _write_scp(task_dir / "hyp.scp", ["utt1 h1"])

    cfg = OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "metrics": [
                {
                    "metric": {"_target_": f"{__name__}.DummyMetric"},
                    "inputs": {"ref": "ref", "hyp": "hyp"},
                }
            ],
        }
    )

    assert _resolve_test_sets(cfg) == ["test_a", "test_b"]
    results = measure(cfg, _INFERENCE_CFG)

    expected_key = get_class_path(DummyMetric())
    assert set(results[expected_key]) == {"test_a", "test_b"}
    assert results[expected_key]["test_a"] == {"count": 1}
    assert results[expected_key]["test_b"] == {"count": 1}


def test_resolve_test_sets_prefers_metrics_config_dataset_over_inference_dir(tmp_path):
    inference_dir = tmp_path / "infer"
    for test_name in ("from_dir_a", "from_dir_b"):
        (inference_dir / test_name).mkdir(parents=True)

    cfg = OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "dataset": {"test": [{"name": "from_cfg"}]},
            "metrics": [{"metric": {"_target_": f"{__name__}.DummyMetric"}}],
        }
    )

    assert _resolve_test_sets(cfg) == ["from_cfg"]


def test_resolve_test_sets_ignores_hidden_directories(tmp_path):
    inference_dir = tmp_path / "infer"
    (inference_dir / ".ignored").mkdir(parents=True)
    (inference_dir / "test_visible").mkdir(parents=True)

    cfg = OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "metrics": [{"metric": {"_target_": f"{__name__}.DummyMetric"}}],
        }
    )

    assert _resolve_test_sets(cfg) == ["test_visible"]


def test_metric_rejects_non_metric_instance(tmp_path):
    cfg = OmegaConf.create(
        {
            "inference_dir": str(tmp_path),
            "dataset": {"test": [{"name": "test_a"}]},
            "metrics": [{"metric": {"_target_": f"{__name__}.NotMetric"}}],
        }
    )

    with pytest.raises(TypeError, match="not a valid BaseMetric instance"):
        measure(cfg, _INFERENCE_CFG)


def test_metric_requires_inputs_in_config(tmp_path):
    cfg = OmegaConf.create(
        {
            "inference_dir": str(tmp_path),
            "dataset": {"test": [{"name": "test_a"}]},
            "metrics": [{"metric": {"_target_": f"{__name__}.NoKeyMetric"}}],
        }
    )

    with pytest.raises(ValueError, match="bind them in the metrics config"):
        measure(cfg, _INFERENCE_CFG)


def test_metric_requires_test_sets_from_config_or_inference_dir(tmp_path):
    cfg = OmegaConf.create(
        {
            "inference_dir": str(tmp_path / "infer"),
            "metrics": [{"metric": {"_target_": f"{__name__}.DummyMetric"}}],
        }
    )

    (tmp_path / "infer").mkdir()
    with pytest.raises(ValueError, match="No test sets found"):
        measure(cfg, _INFERENCE_CFG)


# ---------------------------------------------------------------------------
# declared metric contract, wired through measure()
# ---------------------------------------------------------------------------


class SkipsSuperInitMetric(BaseMetric):
    """Declares nothing, and skips super().__init__() - the gap measure() closes."""

    def __init__(self):
        pass

    def __call__(self, data, test_name, inference_dir):
        return {}


def test_measure_rejects_a_metric_that_skipped_the_init_check(tmp_path):
    inference_dir = tmp_path / "infer"
    test_name = "test-clean"
    (inference_dir / test_name).mkdir(parents=True)

    cfg = OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "dataset": {"test": [{"name": test_name}]},
            "metrics": [{"metric": {"_target_": f"{__name__}.SkipsSuperInitMetric"}}],
        }
    )

    # construction alone does not catch it: __init__ never called super().__init__()
    assert SkipsSuperInitMetric()

    with pytest.raises(TypeError, match="does not declare"):
        measure(cfg, _INFERENCE_CFG)


def test_measure_rejects_metric_whose_input_is_not_a_declared_output(tmp_path):
    inference_dir = tmp_path / "infer"
    test_name = "test-clean"
    (inference_dir / test_name).mkdir(parents=True)

    not_declared = OmegaConf.create({"model": {"_target_": "builtins.dict"}})
    cfg = OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "dataset": {"test": [{"name": test_name}]},
            "metrics": [
                {
                    "metric": {"_target_": f"{__name__}.DummyMetric"},
                    "inputs": {"ref": "ref", "hyp": "hyp"},
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="declares no outputs"):
        measure(cfg, not_declared)


def test_measure_succeeds_for_declared_metric_with_matching_output(tmp_path):
    inference_dir = tmp_path / "infer"
    test_name = "test-clean"
    task_dir = inference_dir / test_name
    task_dir.mkdir(parents=True)
    _write_scp(task_dir / "ref.scp", ["utt1 r1"])
    _write_scp(task_dir / "hyp.scp", ["utt1 h1"])

    cfg = OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "dataset": {"test": [{"name": test_name}]},
            "metrics": [
                {
                    "metric": {"_target_": f"{__name__}.DummyMetric"},
                    "inputs": {"ref": "ref", "hyp": "hyp"},
                }
            ],
        }
    )

    results = measure(cfg, _INFERENCE_CFG)

    expected_key = get_class_path(DummyMetric())
    assert results[expected_key][test_name] == {"count": 1}


class BadOutputMetric(BaseMetric):
    """A declared metric whose __call__ breaks its own output contract."""

    inputs = (Field("ref", "text"), Field("hyp", "text"))
    outputs = (Field("X", "number"),)

    def __call__(self, data, test_name, inference_dir):
        return {"X": "not-a-number"}


def test_measure_rejects_non_number_metric_output(tmp_path):
    inference_dir = tmp_path / "infer"
    test_name = "test-clean"
    task_dir = inference_dir / test_name
    task_dir.mkdir(parents=True)
    _write_scp(task_dir / "ref.scp", ["utt1 r1"])
    _write_scp(task_dir / "hyp.scp", ["utt1 h1"])

    cfg = OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "dataset": {"test": [{"name": test_name}]},
            "metrics": [
                {
                    "metric": {"_target_": f"{__name__}.BadOutputMetric"},
                    "inputs": {"ref": "ref", "hyp": "hyp"},
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="number"):
        measure(cfg, _INFERENCE_CFG)


def test_measure_succeeds_for_wer_against_a_declared_inference_model(tmp_path):
    from espnet3.systems.esp2_asr.metrics.wer import WER

    try:
        import jiwer  # noqa: F401
    except ImportError:
        pytest.skip("jiwer not installed")

    inference_dir = tmp_path / "infer"
    test_name = "test-clean"
    task_dir = inference_dir / test_name
    task_dir.mkdir(parents=True)
    _write_scp(task_dir / "ref.scp", ["utt1 hello world"])
    _write_scp(task_dir / "hyp.scp", ["utt1 hello world"])

    cfg = OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "dataset": {"test": [{"name": test_name}]},
            "metrics": [
                {
                    "metric": {"_target_": "espnet3.systems.esp2_asr.metrics.wer.WER"},
                    "inputs": {"ref": "ref", "hyp": "hyp"},
                }
            ],
        }
    )

    results = measure(cfg, _INFERENCE_CFG)

    expected_key = get_class_path(WER())
    assert results[expected_key][test_name] == {"WER": 0.0}


class _DatasetTextProvider:
    """A minimal provider whose test set has only a `text` column."""

    @staticmethod
    def build_dataset(config):
        return [{"utt_id": "utt1", "text": "hello world"}]

    @staticmethod
    def build_model(config):
        raise NotImplementedError


def test_measure_succeeds_for_wer_with_dataset_and_scp_inputs(tmp_path):
    """A WER scored through measure() with a mixed-source inputs: mapping.

    One declared name binds to a dataset column, the other to an
    inference-written .scp; this drives the full measure() path -
    check_metric_inputs, then the actual scoring - not just
    check_metric_inputs in isolation.
    """
    from espnet3.systems.esp2_asr.metrics.wer import WER

    try:
        import jiwer  # noqa: F401
    except ImportError:
        pytest.skip("jiwer not installed")

    inference_dir = tmp_path / "infer"
    test_name = "test-clean"
    task_dir = inference_dir / test_name
    task_dir.mkdir(parents=True)
    _write_scp(task_dir / "text.scp", ["utt1 hello world"])

    inference_cfg = OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "model": {"_target_": f"{__name__}._DummyInference"},
            "provider": {"_target_": f"{__name__}._DatasetTextProvider"},
        }
    )
    metrics_cfg = OmegaConf.create(
        {
            "inference_dir": str(inference_dir),
            "dataset": {"test": [{"name": test_name}]},
            "metrics": [
                {
                    "metric": {"_target_": "espnet3.systems.esp2_asr.metrics.wer.WER"},
                    "inputs": {"ref": "dataset:text", "hyp": "text"},
                }
            ],
        }
    )

    results = measure(metrics_cfg, inference_config=inference_cfg)

    expected_key = get_class_path(WER())
    assert results[expected_key][test_name] == {"WER": 0.0}
