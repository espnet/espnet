#!/usr/bin/env python3
import importlib.machinery as imm
import importlib.util as iu
import logging
import pathlib
import re
import os

import configargparse


class ModuleInfo:
    def __init__(self, path):
        self.path = pathlib.Path(path)
        name = str(self.path.parent / self.path.stem)
        name = name.replace("/", ".")
        self.name = re.sub(r"^[\.]+", "", name)
        self.module = imm.SourceFileLoader(self.name, path).load_module()
        if not hasattr(self.module, "get_parser"):
            raise ValueError(f"{path} does not have get_parser()")


# parser
parser = configargparse.ArgumentParser(
    description='generate rst file from argparse options',
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('src', type=str, nargs='+',
                    help='source python files that contain get_parser() func')
args = parser.parse_args()


modinfo = []

for p in args.src:
    if "__init__.py" in p:
        continue
    modinfo.append(ModuleInfo(p))


# print refs
for m in modinfo:
    logging.info(f"processing: {m.path.name}")
    d = m.module.get_parser().description
    assert d is not None
    print(f"- :ref:`{m.path.name}`: {d}")

print()

# print argparse
for m in modinfo:
    cmd = m.path.name
    sep = "~" * len(cmd)
    print(f"""

.. _{cmd}:

{cmd}
{sep}

.. argparse::
   :module: {m.name}
   :func: get_parser
   :prog: {cmd}

""")
