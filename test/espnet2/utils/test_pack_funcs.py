from pathlib import Path
import tarfile

import pytest
import yaml

from espnet2.utils.pack_funcs import default_tarinfo
from espnet2.utils.pack_funcs import find_path_and_change_it_recursive
from espnet2.utils.pack_funcs import pack
from espnet2.utils.pack_funcs import unpack


def test_find_path_and_change_it_recursive():
    target = {"a": ["foo/path.npy"], "b": 3}
    target = find_path_and_change_it_recursive(target, "foo/path.npy", "bar/path.npy")
    assert target == {"a": ["bar/path.npy"], "b": 3}


def test_default_tarinfo():
    # Just call
    default_tarinfo("aaa")


def test_pack_unpack(tmp_path: Path):
    files = {"abc.pth": str(tmp_path / "foo.pth")}
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
        yaml_files={"def.yaml": str(tmp_path / "bar.yaml")},
        option=[tmp_path / "a", tmp_path / "b" / "a"],
        outpath=str(tmp_path / "out.tgz"),
    )

    retval = unpack(str(tmp_path / "out.tgz"), str(tmp_path))
    assert retval == {
        "abc": str(tmp_path / "packed" / "abc.pth"),
        "def": str(tmp_path / "packed" / "def.yaml"),
        "option": [
            str(tmp_path / "packed" / "option" / "a"),
            str(tmp_path / "packed" / "option" / "a.1"),
        ],
        "meta": str(tmp_path / "packed" / "meta.yaml"),
    }


def test_pack_not_exist_file():
    with pytest.raises(FileNotFoundError):
        pack(files={"a": "aaa"}, yaml_files={}, outpath="out")


def test_unpack_no_meta_yaml(tmp_path: Path):
    with tarfile.open(tmp_path / "a.tgz", "w:gz"):
        pass
    with pytest.raises(RuntimeError):
        unpack(str(tmp_path / "a.tgz"), "out")
