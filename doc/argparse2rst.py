#!/usr/bin/env python3
import importlib.machinery as imm
import logging
import pathlib
import re
import os
import subprocess

import configargparse


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


class ModuleInfo:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        name = str(self.path.parent / self.path.stem)
        name = name.replace("/", ".")
        self.name = re.sub(r"^[\.]+", "", name)
        self.module = imm.SourceFileLoader(self.name, path).load_module()
        if not hasattr(self.module, "get_parser"):
            raise ValueError(f"{path} does not have get_parser()")


def get_parser():
    parser = configargparse.ArgumentParser(
        description="generate RST from argparse options",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Module Reference",
        help="title for the generated RST",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="output directory to save generated RSTs",
    )
    parser.add_argument(
        "src",
        type=str,
        nargs="+",
        help="source python files that contain get_parser() func",
    )
    return parser


if __name__ == "__main__":
    # parser
    args = get_parser().parse_args()

    modinfo = []
    for p in args.src:
        if "__init__.py" in p:
            continue
        try:
            modinfo.append(ModuleInfo(p))
        except Exception as e:
            logging.error(f"Error processing {p}: {str(e)}")

    print(f"""
{args.title}
{"=" * len(args.title)}

""")

    for m in modinfo:
        logging.info(f"processing: {m.path.name}")
        d = m.module.get_parser().description
        assert d is not None
        print(f"- :ref:`{m.path.name}`: {d}")

    print()

    os.makedirs(args.output_dir, exist_ok=True)

    # print argparse to each files
    for m in modinfo:
        cmd = m.path.name
        sourceurl = "https://github.com/espnet/espnet/blob/" \
            + get_git_revision_hash() + "/" + str(m.path.parent / m.path.stem) + ".py"
        sep = "~" * len(cmd)
        mname = m.name if m.name.startswith("espnet") \
            else ".".join(m.name.split(".")[1:])
        with open(f"{args.output_dir}/{cmd[:-3]}.rst", "w") as writer:  # remove .py
            writer.write(f""".. _{cmd}
{cmd}
{sep}

`source <{sourceurl}>`_

.. argparse::
   :module: {mname}
   :func: get_parser
   :prog: {cmd}

""")
