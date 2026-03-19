from pathlib import Path

import pytest

from espnet3.components.metrics.base_metric import BaseMetric


class DummyMetric(BaseMetric):
    def __call__(self, data, test_name, output_dir):
        return {"count": sum(1 for _ in self.iter_inputs(data, "ref"))}


def _build_inputs(tmp_path: Path, **entries: list[str]) -> dict[str, Path]:
    data = {}
    for key, lines in entries.items():
        path = tmp_path / f"{key}.scp"
        path.write_text("\n".join(lines), encoding="utf-8")
        data[key] = path
    return data


def test_iter_inputs_requires_at_least_one_key(tmp_path: Path):
    metric = DummyMetric()
    data = _build_inputs(tmp_path, ref=["utt1 value1"])

    with pytest.raises(AssertionError, match="At least one SCP key is required"):
        list(metric.iter_inputs(data))


def test_iter_inputs_reads_single_key(tmp_path: Path):
    metric = DummyMetric()
    data = _build_inputs(tmp_path, ref=["utt1 value1", "utt2 value2"])

    assert list(metric.iter_inputs(data, "ref")) == [
        ("utt1", {"ref": "value1"}),
        ("utt2", {"ref": "value2"}),
    ]


def test_iter_inputs_reads_aligned_multiple_keys(tmp_path: Path):
    metric = DummyMetric()
    data = _build_inputs(
        tmp_path,
        ref=["utt1 hello world", "utt2 foo"],
        hyp=["utt1 hello word", "utt2 foo"],
    )

    assert list(metric.iter_inputs(data, "ref", "hyp")) == [
        ("utt1", {"ref": "hello world", "hyp": "hello word"}),
        ("utt2", {"ref": "foo", "hyp": "foo"}),
    ]


def test_iter_inputs_rejects_length_mismatch(tmp_path: Path):
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


def test_iter_inputs_rejects_utt_id_mismatch(tmp_path: Path):
    metric = DummyMetric()
    data = _build_inputs(
        tmp_path,
        ref=["utt1 r1"],
        hyp=["utt2 h1"],
    )

    with pytest.raises(AssertionError, match="UID mismatch between ref and hyp"):
        list(metric.iter_inputs(data, "ref", "hyp"))


def test_iter_inputs_skips_blank_lines_and_keeps_empty_values(tmp_path: Path):
    metric = DummyMetric()
    data = _build_inputs(
        tmp_path,
        ref=["utt1 value1", "", "utt2"],
    )

    assert list(metric.iter_inputs(data, "ref")) == [
        ("utt1", {"ref": "value1"}),
        ("utt2", {"ref": ""}),
    ]


def test_iter_scp_file_splits_only_on_first_whitespace():
    metric = DummyMetric()
    rows = list(metric._iter_scp_file(["utt1 many words here", "utt2 single"]))

    assert rows == [
        ("utt1", "many words here"),
        ("utt2", "single"),
    ]
