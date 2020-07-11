from collections import defaultdict
from datetime import datetime
from io import BytesIO
from io import TextIOWrapper
import os
from pathlib import Path
import sys
import tarfile
from typing import Dict
from typing import Iterable
from typing import Union
import zipfile

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

    def add(self, filename, arcname=None):
        if self.type == "tar":
            return self.fopen.add(filename, arcname)
        elif self.type == "zip":
            return self.fopen.write(filename, arcname)
        else:
            raise ValueError(f"Not supported: type={self.type}")

    def addfile(self, info, fileobj):
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
            return self.fopen.extract(info, path)
        elif self.type == "zip":
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


def unpack(
    input_tarfile: Union[Path, str], outpath: Union[Path, str],
) -> Dict[str, str]:
    """Scan all files in the archive file and return as a dict of files.

    Examples:
        tarfile:
           packed/asr_model_file.pth
           packed/option/some1.file
           packed/option/some2.file

        >>> unpack("tarfile", "out")
        {'asr_model_file': 'out/packed/asr_model_file.pth',
         'option': ['out/packed/option/some1.file', 'out/packed/option/some2.file']}
    """
    input_tarfile = Path(input_tarfile)
    outpath = Path(outpath)

    with Archiver(input_tarfile) as archive:
        for info in archive:
            if Path(archive.get_name_from_info(info)).name == "meta.yaml":
                d = yaml.safe_load(archive.extractfile(info))
                yaml_files = d["yaml_files"]
                break
        else:
            raise RuntimeError("Format error: not found meta.yaml")

        retval = defaultdict(list)
        for info in archive:
            outname = outpath / archive.get_name_from_info(info)
            outname.parent.mkdir(parents=True, exist_ok=True)
            if archive.get_name_from_info(info) in yaml_files:
                d = yaml.safe_load(archive.extractfile(info))
                # Rewrite yaml
                for info2 in archive:
                    name = archive.get_name_from_info(info2)
                    d = find_path_and_change_it_recursive(d, name, str(outpath / name))
                with outname.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(d, f)
            else:
                archive.extract(info, path=outpath)

            key = archive.get_name_from_info(info).split("/")[1]
            key = Path(key).stem
            retval[key].append(str(outname))
        retval = {k: v[0] if len(v) == 1 else v for k, v in retval.items()}
        return retval


def pack(
    files: Dict[str, Union[str, Path]],
    yaml_files: Dict[str, Union[str, Path]],
    outpath: Union[str, Path],
    option: Iterable[Union[str, Path]] = (),
    dirname: str = "packed",
):
    for v in list(files.values()) + list(yaml_files.values()) + list(option):
        if not Path(v).exists():
            raise FileNotFoundError(f"No such file or directory: {v}")
    dirname = Path(dirname)

    files_map = {}
    for name, src in list(files.items()):
        # Save as e.g. packed/asr_model_file.pth
        dst = str(dirname / name)
        files_map[dst] = src

    for src in option:
        # Save as packed/option/${basename}
        idx = 0
        while True:
            p = Path(src)
            if idx == 0:
                dst = str(dirname / "option" / p.name)
            else:
                dst = str(dirname / "option" / f"{p.stem}.{idx}{p.suffix}")
            if dst not in files_map:
                files_map[dst] = src
                break
            idx += 1

    # Read yaml and Change the file path to the archived path
    yaml_files_map = {}
    for name, path in yaml_files.items():
        with open(path, "r", encoding="utf-8") as f:
            dic = yaml.safe_load(f)
            for dst, src in files_map.items():
                dic = find_path_and_change_it_recursive(dic, src, dst)
            dst = str(dirname / name)
            yaml_files_map[dst] = dic

    meta_objs = dict(
        files=list(files_map),
        yaml_files=list(yaml_files_map),
        timestamp=datetime.now().timestamp(),
        python=sys.version,
    )

    try:
        import torch

        meta_objs.update(torch=torch.__version__)
    except ImportError:
        pass
    try:
        import espnet

        meta_objs.update(espnet=espnet.__version__)
    except ImportError:
        pass

    Path(outpath).parent.mkdir(parents=True, exist_ok=True)
    with Archiver(outpath, mode="w") as archive:
        # Write packed/meta.yaml
        fileobj = BytesIO(yaml.safe_dump(meta_objs).encode())
        info = archive.generate_info(dirname / "meta.yaml", fileobj.getbuffer().nbytes)
        archive.addfile(info, fileobj=fileobj)

        for dst, dic in yaml_files_map.items():
            # Dump dict as yaml-bytes
            fileobj = BytesIO(yaml.safe_dump(dic).encode())
            # Embed the yaml-bytes in tarfile
            info = archive.generate_info(dst, fileobj.getbuffer().nbytes)
            archive.addfile(info, fileobj=fileobj)
        for dst, src in files_map.items():
            # Resolve to avoid symbolic link
            archive.add(Path(src).resolve(), dst)
