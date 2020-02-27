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

import yaml

DIRNAME = Path("packed")


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


def default_tarinfo(name) -> tarfile.TarInfo:
    """Generate TarInfo using system information"""
    tarinfo = tarfile.TarInfo(str(name))
    if os.name == "posix":
        tarinfo.gid = os.getgid()
        tarinfo.uid = os.getuid()
    tarinfo.mtime = datetime.now().timestamp()
    # Keep mode as default
    return tarinfo


def unpack(
    input_tarfile: Union[Path, str], outpath: Union[Path, str]
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

    with tarfile.open(input_tarfile) as tar:
        for tarinfo in tar:
            if tarinfo.name == str(DIRNAME / "meta.yaml"):
                d = yaml.safe_load(TextIOWrapper(tar.extractfile(tarinfo)))
                yaml_files = d["yaml_files"]
                break
        else:
            raise RuntimeError("Format error: not found meta.yaml")

        retval = defaultdict(list)
        for tarinfo in tar:
            outname = outpath / tarinfo.name
            outname.parent.mkdir(parents=True, exist_ok=True)
            if tarinfo.name in yaml_files:
                d = yaml.safe_load(TextIOWrapper(tar.extractfile(tarinfo)))
                # Rewrite yaml
                for tarinfo2 in tar:
                    d = find_path_and_change_it_recursive(
                        d, tarinfo2.name, str(outpath / tarinfo2.name)
                    )
                with outname.open("w", encoding="utf-8") as f:
                    yaml.safe_dump(d, f)
            else:
                tar.extract(tarinfo, path=outpath)

            key = tarinfo.name.split("/")[1]
            key = Path(key).stem
            retval[key].append(str(outname))
        retval = {k: v[0] if len(v) == 1 else v for k, v in retval.items()}
        return retval


def pack(
    files: Dict[str, Union[str, Path]],
    yaml_files: Dict[str, Union[str, Path]],
    outpath: Union[str, Path],
    option: Iterable[Union[str, Path]] = (),
    mode: str = "w:gz",
):
    for v in list(files.values()) + list(yaml_files.values()) + list(option):
        if not Path(v).exists():
            raise FileNotFoundError(f"No such file or directory: {v}")

    files_map = {}
    for name, src in list(files.items()):
        # Save as e.g. packed/asr_model_file.pth
        dst = str(DIRNAME / name)
        files_map[dst] = src

    for src in option:
        # Save as packed/option/${basename}
        idx = 0
        while True:
            p = Path(src)
            if idx == 0:
                dst = str(DIRNAME / "option" / p.name)
            else:
                dst = str(DIRNAME / "option" / f"{p.stem}.{idx}{p.suffix}")
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
            dst = str(DIRNAME / name)
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
    with tarfile.open(outpath, mode=mode) as tar:
        # Write packed/meta.yaml
        fileobj = BytesIO(yaml.safe_dump(meta_objs).encode())
        tarinfo = default_tarinfo(DIRNAME / "meta.yaml")
        tarinfo.size = fileobj.getbuffer().nbytes
        tar.addfile(tarinfo, fileobj=fileobj)

        for dst, dic in yaml_files_map.items():
            # Dump dict as yaml-bytes
            fileobj = BytesIO(yaml.safe_dump(dic).encode())
            # Embed the yaml-bytes in tarfile
            tarinfo = default_tarinfo(dst)
            tarinfo.size = fileobj.getbuffer().nbytes
            tar.addfile(tarinfo, fileobj=fileobj)
        for dst, src in files_map.items():
            # Resolve to avoid symbolic link
            tar.add(Path(src).resolve(), dst)
