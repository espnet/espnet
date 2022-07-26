import tarfile
from pathlib import Path

import pytest
import yaml

from espnet2.main_funcs.pack_funcs import (
    find_path_and_change_it_recursive,
    pack,
    unpack,
)


def test_find_path_and_change_it_recursive():
    target = {"a": ["foo/path.npy"], "b": 3}
    target = find_path_and_change_it_recursive(target, "foo/path.npy", "bar/path.npy")
    assert target == {"a": ["bar/path.npy"], "b": 3}


@pytest.mark.parametrize(
    "type",
    ["tgz", "tar", "tbz2", "txz", "zip"],
)
def test_pack_unpack(tmp_path: Path, type):
    files = {"abc": str(tmp_path / "foo.pth")}
    with (tmp_path / "foo.pth").open("w"):
        pass
    with (tmp_path / "bar.yaml").open("w") as f:
        # I dared to stack "/" to test
        yaml.safe_dump({"a": str(tmp_path / "//foo.pth")}, f)
    with (tmp_path / "a").open("w"):
        pass
    (tmp_path / "b").mkdir(parents=True, exist_ok=True)
    with (tmp_path / "b" / "a").open("w"):
        pass

    pack(
        files=files,
        yaml_files={"def": str(tmp_path / "bar.yaml")},
        option=[tmp_path / "a", tmp_path / "b" / "a"],
        outpath=str(tmp_path / f"out.{type}"),
    )

    retval = unpack(str(tmp_path / f"out.{type}"), str(tmp_path))
    # Retry unpack. If cache file exists, generate dict from it
    retval2 = unpack(str(tmp_path / f"out.{type}"), str(tmp_path))
    assert retval == {
        "abc": str(tmp_path / tmp_path / "foo.pth"),
        "def": str(tmp_path / tmp_path / "bar.yaml"),
    }
    assert retval2 == {
        "abc": str(tmp_path / tmp_path / "foo.pth"),
        "def": str(tmp_path / tmp_path / "bar.yaml"),
    }


def test_pack_not_exist_file():
    with pytest.raises(FileNotFoundError):
        pack(files={"a": "aaa"}, yaml_files={}, outpath="out")


def test_unpack_no_meta_yaml(tmp_path: Path):
    with tarfile.open(tmp_path / "a.tgz", "w:gz"):
        pass
    with pytest.raises(RuntimeError):
        unpack(str(tmp_path / "a.tgz"), "out")


@pytest.mark.parametrize(
    "type",
    ["tgz", "tar", "tbz2", "txz", "zip"],
)
def test_pack_unpack_recursive(tmp_path: Path, type):
    p = tmp_path / "a" / "b"
    p.mkdir(parents=True)
    with (p / "foo.pth").open("w"):
        pass

    pack(
        files={},
        yaml_files={},
        option=[p],
        outpath=str(tmp_path / f"out.{type}"),
    )

    unpack(str(tmp_path / f"out.{type}"), str(tmp_path))
    assert (tmp_path / p / "foo.pth").exists()
