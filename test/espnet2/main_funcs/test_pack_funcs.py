import tarfile
from pathlib import Path

import pytest
import yaml

from espnet2.main_funcs.pack_funcs import (
    UnsafeArchiveMemberError,
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


def _tar_with_members(path: Path, members):
    """Write a tar whose meta.yaml declares `yaml_files` / `files` as given."""
    import io

    with tarfile.open(path, "w") as tf:
        for name, data in members:
            info = tarfile.TarInfo(name)
            info.size = len(data)
            tf.addfile(info, io.BytesIO(data))


@pytest.mark.parametrize("member", ["../escaped.pth", "a/../../escaped.pth"])
def test_unpack_refuses_tar_slip_member(tmp_path: Path, member):
    """A tar member whose name escapes outpath must not be written."""
    meta = yaml.safe_dump({"yaml_files": {}, "files": {}}).encode()
    _tar_with_members(
        tmp_path / "evil.tar",
        [("meta.yaml", meta), (member, b"PWNED")],
    )
    outpath = tmp_path / "deep" / "out"
    outpath.mkdir(parents=True)

    with pytest.raises(UnsafeArchiveMemberError):
        unpack(str(tmp_path / "evil.tar"), str(outpath))
    assert not (tmp_path / "deep" / "escaped.pth").exists()
    assert not (tmp_path / "escaped.pth").exists()


def test_unpack_refuses_traversal_in_yaml_rewrite_branch(tmp_path: Path):
    """The yaml-rewrite branch joins paths by hand, bypassing tarfile."""
    meta = yaml.safe_dump(
        {"yaml_files": {"cfg": "../escaped.yaml"}, "files": {}}
    ).encode()
    _tar_with_members(
        tmp_path / "evil.tar",
        [
            ("meta.yaml", meta),
            ("../escaped.yaml", yaml.safe_dump({"a": 1}).encode()),
        ],
    )
    outpath = tmp_path / "deep" / "out"
    outpath.mkdir(parents=True)

    with pytest.raises(UnsafeArchiveMemberError):
        unpack(str(tmp_path / "evil.tar"), str(outpath))
    assert not (tmp_path / "deep" / "escaped.yaml").exists()


def test_unpack_refuses_absolute_member(tmp_path: Path):
    """An absolute member name replaces outpath under pathlib semantics."""
    escaped = tmp_path / "absolute_escape.pth"
    meta = yaml.safe_dump({"yaml_files": {}, "files": {}}).encode()
    with tarfile.open(tmp_path / "evil.tar", "w") as tf:
        import io

        info = tarfile.TarInfo("meta.yaml")
        info.size = len(meta)
        tf.addfile(info, io.BytesIO(meta))
        # tarfile.add() strips leading "/", so set the name on TarInfo directly.
        info2 = tarfile.TarInfo(str(escaped))
        info2.size = 5
        tf.addfile(info2, io.BytesIO(b"PWNED"))

    outpath = tmp_path / "out"
    outpath.mkdir()
    with pytest.raises(UnsafeArchiveMemberError):
        unpack(str(tmp_path / "evil.tar"), str(outpath))
    assert not escaped.exists()
