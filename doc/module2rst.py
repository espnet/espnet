#!/usr/bin/env python3
from glob import glob
import importlib
import os

import configargparse


# parser
parser = configargparse.ArgumentParser(
    description='generate RST files from <root> module recursively into <dst>/_gen',
    config_file_parser_class=configargparse.YAMLConfigFileParser,
    formatter_class=configargparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('root', type=str,
                    help='root module to generate docs recursively')
parser.add_argument('dst', type=str,
                    help='destination path to generate RSTs')
parser.add_argument('--exclude', nargs='*', default=[],
                    help='exclude module name')
args = parser.parse_args()
print(args)


def to_module(path_name):
    ret = path_name.replace(".py", "").replace("/", ".")
    if ret.endswith("."):
        return ret[:-1]
    return ret


def gen_rst(module_path, f):
    name = to_module(module_path)
    module = importlib.import_module(name)
    title = name + " package"
    sep = "=" * len(title)
    doc = module.__doc__
    if doc is None:
        doc = ""
    f.write(f"""
{title}
{sep}
{doc}

""")

    for cpath in glob(module_path + "/**/*.py", recursive=True):
        print(cpath)
        if not os.path.exists(cpath):
            continue
        if "__pycache__" in cpath:
            continue
        cname = to_module(cpath)
        csep = "-" * len(cname)
        f.write(f"""
.. _{cname}:

{cname}
{csep}

.. automodule:: {cname}
    :members:
    :undoc-members:
    :show-inheritance:

""")
    f.flush()


modules_rst = """
.. toctree::
   :maxdepth: 1
   :caption: Package Reference:

"""
gendir = args.dst + "/_gen"
os.makedirs(gendir, exist_ok=True)
for p in glob(args.root + "/**", recursive=False):
    if p in args.exclude:
        continue
    if "__pycache__" in p:
        continue
    if "__init__" in p:
        continue
    fname = to_module(p) + ".rst"
    dst = f"{gendir}/{fname}"
    modules_rst += f"   ./_gen/{fname}\n"
    print(f"[INFO] generating {dst}")
    with open(dst, "w") as f:
        gen_rst(p, f)


with open(gendir + "/modules.rst", "w") as f:
    f.write(modules_rst)
