import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from espnet3.components.metrics.base_metric import BaseMetric
from espnet3.systems.base.metric import _resolve_test_sets, measure
from espnet3.utils.config_utils import load_config_with_defaults
from espnet3.utils.scp_utils import get_class_path

TEMPLATE_METRICS_CONFIG = (
    Path(__file__).resolve().parents[4] / "egs3/TEMPLATE/asr/conf/metrics.yaml"
)


class DummyMetric(BaseMetric):
    ref_key = "ref"
    hyp_key = "hyp"

    def __call__(self, data, test_name, inference_dir):
        return {"count": sum(1 for _ in self.iter_inputs(data, "ref"))}


class NoKeyMetric(BaseMetric):
    def __call__(self, data, test_name, inference_dir):
        return {"ok": True}


class PathMetric(BaseMetric):
    ref_key = "ref"
    hyp_key = "hyp"

    def __call__(self, data, test_name, inference_dir):
        task_dir = Path(inference_dir) / test_name
        assert data["ref"] == task_dir / "ref.scp"
        assert [row["hyp"] for _, row in self.iter_inputs(data, "hyp")] == ["h1"]
        return {"ok": True}


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


def test_metric_uses_metric_keys_and_writes_json(tmp_path):
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
            "metrics": [{"metric": {"_target_": f"{__name__}.DummyMetric"}}],
        }
    )

    results = measure(cfg)

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

    results = measure(cfg)

    expected_key = get_class_path(NoKeyMetric())
    assert results[expected_key][test_name] == {"ok": True}


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
            "metrics": [{"metric": {"_target_": f"{__name__}.PathMetric"}}],
        }
    )

    results = measure(cfg)

    expected_key = get_class_path(PathMetric())
    assert results[expected_key][test_name] == {"ok": True}


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
            "metrics": [{"metric": {"_target_": f"{__name__}.DummyMetric"}}],
        }
    )

    assert _resolve_test_sets(cfg) == ["test_a", "test_b"]
    results = measure(cfg)

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
        measure(cfg)


def test_metric_requires_inputs_when_metric_has_no_keys(tmp_path):
    cfg = OmegaConf.create(
        {
            "inference_dir": str(tmp_path),
            "dataset": {"test": [{"name": "test_a"}]},
            "metrics": [{"metric": {"_target_": f"{__name__}.NoKeyMetric"}}],
        }
    )

    with pytest.raises(ValueError, match="requires inputs in config"):
        measure(cfg)


def test_metric_requires_test_sets_from_config_or_inference_dir(tmp_path):
    cfg = OmegaConf.create(
        {
            "inference_dir": str(tmp_path / "infer"),
            "metrics": [{"metric": {"_target_": f"{__name__}.DummyMetric"}}],
        }
    )

    (tmp_path / "infer").mkdir()
    with pytest.raises(ValueError, match="No test sets found"):
        measure(cfg)


@pytest.mark.execution_timeout(30)
def test_measure_scores_real_wer_and_cer_from_template_metrics_config(tmp_path):
    """e2e: the shipped TEMPLATE metrics.yaml drives real WER/CER scoring.

    Regression coverage for metrics#13: test_metric.py previously only used
    DummyMetric, never the real config shipped to recipes nor the real
    jiwer-backed WER/CER implementations.
    """
    try:
        import jiwer  # noqa: F401
    except ImportError:
        pytest.skip("jiwer is required for this test")

    inference_dir = tmp_path / "infer"
    test_name = "test_a"
    task_dir = inference_dir / test_name
    task_dir.mkdir(parents=True)
    _write_scp(task_dir / "ref.scp", ["utt1 hello world", "utt2 abc"])
    _write_scp(task_dir / "hyp.scp", ["utt1 hello word", "utt2 axc"])

    cfg = load_config_with_defaults(str(TEMPLATE_METRICS_CONFIG))
    cfg.exp_dir = str(tmp_path / "exp")
    cfg.inference_dir = str(inference_dir)
    cfg.dataset = {"test": [{"name": test_name}]}

    results = measure(cfg)

    from espnet3.systems.asr.metrics.cer import CER
    from espnet3.systems.asr.metrics.wer import WER

    assert results[get_class_path(WER())][test_name] == {"WER": 66.67}
    assert results[get_class_path(CER())][test_name] == {"CER": 14.29}
    assert (inference_dir / "metrics.json").is_file()


def test_measure_rejects_null_metrics(tmp_path):
    """metrics: null (a valid YAML shape) must raise a clear error.

    Regression: previously ``measure()`` only checked ``hasattr(config,
    "metrics")``, which is True even when the value is ``None``, so this
    case fell through to ``enumerate(None)`` and raised an opaque
    ``TypeError: 'NoneType' object is not iterable``.
    """
    inference_dir = tmp_path / "infer" / "test_a"
    inference_dir.mkdir(parents=True)
    _write_scp(inference_dir / "ref.scp", ["utt1 r1"])
    _write_scp(inference_dir / "hyp.scp", ["utt1 h1"])

    cfg = OmegaConf.create(
        {
            "inference_dir": str(tmp_path / "infer"),
            "dataset": {"test": [{"name": "test_a"}]},
            "metrics": None,
        }
    )

    with pytest.raises(AssertionError, match="Please specify `metrics`"):
        measure(cfg)


def test_measure_duplicate_metric_classes_overwrite_results(tmp_path):
    """Document current behavior: two configs of the same metric class collide.

    ``measure()`` keys ``results`` by ``get_class_path(metric)``, so two
    entries for the same class (e.g. CER with different ``clean_types``)
    silently overwrite each other rather than raising or being kept
    separately. This is a known gap (metrics#13 outlook), recorded here as a
    regression test on the current behavior rather than a design change.
    """
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
                {"metric": {"_target_": f"{__name__}.DummyMetric"}},
                {"metric": {"_target_": f"{__name__}.DummyMetric"}},
            ],
        }
    )

    results = measure(cfg)

    # Only one entry survives even though two metric configs were provided.
    assert list(results.keys()) == [get_class_path(DummyMetric())]
