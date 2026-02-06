from pathlib import Path

import pytest

from espnet3.utils.scp_utils import get_cls_path, load_scp_fields


def test_get_cls_path_reports_module_and_class():
    class Dummy:
        pass

    dummy = Dummy()
    path = get_cls_path(dummy)
    assert path.endswith(".Dummy")
    assert "Dummy" in path


def test_load_scp_fields_reads_and_aligns(tmp_path: Path):
    inference_dir = tmp_path / "exp" / "infer"
    test_name = "test-clean"
    task_dir = inference_dir / test_name
    task_dir.mkdir(parents=True, exist_ok=True)

    (task_dir / "ref.scp").write_text("utt2 ref2\nutt1 ref1\n", encoding="utf-8")
    (task_dir / "hyp.scp").write_text("utt1 hyp1\nutt2 hyp2\n", encoding="utf-8")

    data = load_scp_fields(
        inference_dir=inference_dir,
        test_name=test_name,
        inputs={"ref": "ref", "hyp": "hyp"},
    )

    assert data["utt_id"] == ["utt1", "utt2"]
    assert data["ref"] == ["ref1", "ref2"]
    assert data["hyp"] == ["hyp1", "hyp2"]


def test_load_scp_fields_missing_file_raises(tmp_path: Path):
    inference_dir = tmp_path
    (inference_dir / "test-other").mkdir(parents=True, exist_ok=True)

    with pytest.raises(AssertionError):
        load_scp_fields(
            inference_dir=inference_dir, test_name="test-other", inputs=["ref"]
        )
