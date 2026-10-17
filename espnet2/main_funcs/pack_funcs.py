import os
import sys
import tarfile
import zipfile
from datetime import datetime
from io import BytesIO, TextIOWrapper
from pathlib import Path
from typing import Dict, Iterable, Optional, Union

import yaml


class Archiver:
    def __init__(self, file, mode="r"):
        if Path(file).suffix == ".tar":
            self.type = "tar"
        elif Path(file).suffix == ".tgz" or Path(file).suffixes == [".tar", ".gz"]:
            self.type = "tar"
            if mode == "w":
                mode = "w:gz"
        elif Path(file).suffix == ".tbz2" or Path(file).suffixes == [".tar", ".bz2"]:
            self.type = "tar"
            if mode == "w":
                mode = "w:bz2"
        elif Path(file).suffix == ".txz" or Path(file).suffixes == [".tar", ".xz"]:
            self.type = "tar"
            if mode == "w":
                mode = "w:xz"
        elif Path(file).suffix == ".zip":
            self.type = "zip"
        else:
            raise ValueError(f"Cannot detect archive format: type={file}")

        if self.type == "tar":
            self.fopen = tarfile.open(file, mode=mode)
        elif self.type == "zip":
            self.fopen = zipfile.ZipFile(file, mode=mode)
        else:
            raise ValueError(f"Not supported: type={type}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.fopen.close()

    def close(self):
        self.fopen.close()

    def __iter__(self):
        if self.type == "tar":
            return iter(self.fopen)
        elif self.type == "zip":
            return iter(self.fopen.infolist())
        else:
            raise ValueError(f"Not supported: type={self.type}")

    def add(self, filename, arcname=None, recursive: bool = True):
        if arcname is not None:
            print(f"adding: {arcname}")
        else:
            print(f"adding: {filename}")

        if recursive and Path(filename).is_dir():
            for f in Path(filename).glob("**/*"):
                if f.is_dir():
                    continue

                if arcname is not None:
                    _arcname = Path(arcname) / f
                else:
                    _arcname = None

                self.add(f, _arcname)
            return

        if self.type == "tar":
            return self.fopen.add(filename, arcname)
        elif self.type == "zip":
            return self.fopen.write(filename, arcname)
        else:
            raise ValueError(f"Not supported: type={self.type}")

    def addfile(self, info, fileobj):
        print(f"adding: {self.get_name_from_info(info)}")

        if self.type == "tar":
            return self.fopen.addfile(info, fileobj)
        elif self.type == "zip":
            return self.fopen.writestr(info, fileobj.read())
        else:
            raise ValueError(f"Not supported: type={self.type}")

    def generate_info(self, name, size) -> Union[tarfile.TarInfo, zipfile.ZipInfo]:
        """Generate TarInfo using system information"""
        if self.type == "tar":
            tarinfo = tarfile.TarInfo(str(name))
            if os.name == "posix":
                tarinfo.gid = os.getgid()
                tarinfo.uid = os.getuid()
            tarinfo.mtime = datetime.now().timestamp()
            tarinfo.size = size
            # Keep mode as default
            return tarinfo
        elif self.type == "zip":
            zipinfo = zipfile.ZipInfo(str(name), datetime.now().timetuple()[:6])
            zipinfo.file_size = size
            return zipinfo
        else:
            raise ValueError(f"Not supported: type={self.type}")

    def get_name_from_info(self, info):
        if self.type == "tar":
            assert isinstance(info, tarfile.TarInfo), type(info)
            return info.name
        elif self.type == "zip":
            assert isinstance(info, zipfile.ZipInfo), type(info)
            return info.filename
        else:
            raise ValueError(f"Not supported: type={self.type}")

    def extract(self, info, path=None):
        if self.type == "tar":
            # filter="data" refuses members whose name escapes ``path`` (absolute
            # paths, "..", links pointing outside) and drops device files and
            # setuid/setgid bits. Without it tarfile defaults to "fully_trusted"
            # on Python < 3.14, which allows arbitrary file write (tar-slip).
            return self.fopen.extract(info, path, filter="data")
        elif self.type == "zip":
            # zipfile.extract already sanitizes "..", absolute paths and drive
            # letters, and never materializes symlinks.
            return self.fopen.extract(info, path)
        else:
            raise ValueError(f"Not supported: type={self.type}")

    def extractfile(self, info, mode="r"):
        if self.type == "tar":
            f = self.fopen.extractfile(info)
            if mode == "r":
                return TextIOWrapper(f)
            else:
                return f
        elif self.type == "zip":
            if mode == "rb":
                mode = "r"
            return self.fopen.open(info, mode)
        else:
            raise ValueError(f"Not supported: type={self.type}")


class UnsafeArchiveMemberError(RuntimeError):
    """Raised when an archive member would be written outside the output dir."""


def _resolve_within(outpath: Path, name: str) -> Path:
    """Join ``name`` onto ``outpath`` and refuse to escape it.

    ``tarfile.extract(..., filter="data")`` guards the extraction sink, but
    ``unpack`` also joins member names by hand to rewrite yaml files and to
    build its return value. Those joins bypass tarfile entirely, so they need
    the same check. An absolute ``name`` replaces ``outpath`` under pathlib
    semantics and is rejected here as well.

    Args:
        outpath: Directory the archive is being unpacked into.
        name: Member name taken from the archive or from its ``meta.yaml``.

    Returns:
        Path: ``outpath / name`` unchanged, once it is known to stay at or
        below ``outpath``. The plain join is returned rather than the resolved
        path so that callers keep receiving the relative paths they always did.

    Raises:
        UnsafeArchiveMemberError: If the name resolves outside ``outpath``.
    """
    base = outpath.resolve()
    target = (base / name).resolve()
    if target != base and base not in target.parents:
        raise UnsafeArchiveMemberError(
            f"Refusing to unpack {name!r}: it resolves to {target}, "
            f"which is outside {base}"
        )
    return outpath / name


def find_path_and_change_it_recursive(value, src: str, tgt: str):
    if isinstance(value, dict):
        return {
            k: find_path_and_change_it_recursive(v, src, tgt) for k, v in value.items()
        }
    elif isinstance(value, (list, tuple)):
        return [find_path_and_change_it_recursive(v, src, tgt) for v in value]
    elif isinstance(value, str) and Path(value) == Path(src):
        return tgt
    else:
        return value


def get_dict_from_cache(meta: Union[Path, str]) -> Optional[Dict[str, str]]:
    meta = Path(meta)
    outpath = meta.parent.parent
    if not meta.exists():
        return None

    with meta.open("r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
        assert isinstance(d, dict), type(d)
        yaml_files = d["yaml_files"]
        files = d["files"]
        assert isinstance(yaml_files, dict), type(yaml_files)
        assert isinstance(files, dict), type(files)

        retval = {}
        for key, value in list(yaml_files.items()) + list(files.items()):
            if not (outpath / value).exists():
                return None
            retval[key] = str(outpath / value)
        return retval


def unpack(
    input_archive: Union[Path, str],
    outpath: Union[Path, str],
    use_cache: bool = True,
) -> Dict[str, str]:
    """Scan all files in the archive file and return as a dict of files.

    Examples:
        tarfile:
           model.pth
           some1.file
           some2.file

        >>> unpack("tarfile", "out")
        {'asr_model_file': 'out/model.pth'}
    """
    input_archive = Path(input_archive)
    outpath = Path(outpath)

    with Archiver(input_archive) as archive:
        for info in archive:
            if Path(archive.get_name_from_info(info)).name == "meta.yaml":
                meta_path = _resolve_within(outpath, archive.get_name_from_info(info))
                if use_cache and meta_path.exists():
                    retval = get_dict_from_cache(meta_path)
                    if retval is not None:
                        return retval
                d = yaml.safe_load(archive.extractfile(info))
                assert isinstance(d, dict), type(d)
                yaml_files = d["yaml_files"]
                files = d["files"]
                assert isinstance(yaml_files, dict), type(yaml_files)
                assert isinstance(files, dict), type(files)
                break
        else:
            raise RuntimeError("Format error: not found meta.yaml")

        for info in archive:
            fname = archive.get_name_from_info(info)
            # Checked even for the tarfile branch below: the yaml-rewrite path
            # writes through this join without going through tarfile at all.
            outname = _resolve_within(outpath, fname)
            outname.parent.mkdir(parents=True, exist_ok=True)
            if fname in set(yaml_files.values()):
                d = yaml.safe_load(archive.extractfile(info))
                # Rewrite yaml
                for info2 in archive:
                    name = archive.get_name_from_info(info2)
                    d = find_path_and_change_it_recursive(d, name, str(outpath / name))
                with outname.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(d, f)
            else:
                archive.extract(info, path=outpath)

        retval = {}
        for key, value in list(yaml_files.items()) + list(files.items()):
            # Deliberately NOT passed through _resolve_within: pack() records
            # absolute paths in meta.yaml whenever the packed file lived outside
            # the CWD, so this join legitimately resolves outside outpath for
            # such archives. These are returned strings, not a write sink - the
            # two write sinks above are what the traversal guard protects.
            retval[key] = str(outpath / value)
        return retval


def _to_relative_or_resolve(f):
    # Resolve to avoid symbolic link
    p = Path(f).resolve()
    try:
        # Change to relative if it can
        p = p.relative_to(Path(".").resolve())
    except ValueError:
        pass
    return str(p)


def pack(
    files: Dict[str, Union[str, Path]],
    yaml_files: Dict[str, Union[str, Path]],
    outpath: Union[str, Path],
    option: Iterable[Union[str, Path]] = (),
):
    for v in list(files.values()) + list(yaml_files.values()) + list(option):
        if not Path(v).exists():
            raise FileNotFoundError(f"No such file or directory: {v}")

    files = {k: _to_relative_or_resolve(v) for k, v in files.items()}
    yaml_files = {k: _to_relative_or_resolve(v) for k, v in yaml_files.items()}
    option = [_to_relative_or_resolve(v) for v in option]

    meta_objs = dict(
        files=files,
        yaml_files=yaml_files,
        timestamp=datetime.now().timestamp(),
        python=sys.version,
    )

    try:
        import torch

        meta_objs.update(torch=str(torch.__version__))
    except ImportError:
        pass
    try:
        import espnet2

        meta_objs.update(espnet=espnet2.__version__)
    except ImportError:
        pass

    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with Archiver(outpath, mode="w") as archive:
        # Write packed/meta.yaml
        fileobj = BytesIO(yaml.safe_dump(meta_objs).encode())
        info = archive.generate_info("meta.yaml", fileobj.getbuffer().nbytes)
        archive.addfile(info, fileobj=fileobj)

        for f in list(yaml_files.values()) + list(files.values()) + list(option):
            archive.add(f)

    print(f"Generate: {outpath}")
