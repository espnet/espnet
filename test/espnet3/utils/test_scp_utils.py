from pathlib import Path

import pytest

from espnet3.utils.scp_utils import (
    get_class_path,
    load_scp_paths,
)


def test_get_class_path_reports_module_and_class():
    class Dummy:
        pass

    dummy = Dummy()
    path = get_class_path(dummy)
    assert path.endswith(".Dummy")
    assert "Dummy" in path


def test_load_scp_paths_returns_alias_to_path_mapping(tmp_path: Path):
    inference_dir = tmp_path / "exp" / "infer"
    task_dir = inference_dir / "test-clean"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / "ref.scp").write_text("utt1 ref1\n", encoding="utf-8")

    paths = load_scp_paths(inference_dir, "test-clean", {"ref": "ref"})

    assert paths == {"ref": task_dir / "ref.scp"}


def test_load_scp_paths_missing_file_raises(tmp_path: Path):
    inference_dir = tmp_path
    (inference_dir / "test-other").mkdir(parents=True, exist_ok=True)

    with pytest.raises(AssertionError):
        load_scp_paths(
            inference_dir=inference_dir, test_name="test-other", inputs=["ref"]
        )
