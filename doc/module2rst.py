#!/usr/bin/env python3
import importlib
import logging
import os
import pkgutil

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
args = parser.parse_args()


root_name = args.root
root = importlib.import_module(root_name)
root_list = root.__path__
assert len(root_list) == 1
root_path = root_list[0]


def to_path(module_name):
    assert module_name.startswith(root_name)
    return module_name.replace(".", "/").replace(root_name, root_path)


def children(parent, recursive=False):
    pname = parent.__name__
    for info in pkgutil.iter_modules([to_path(pname)]):
        cname = f"{pname}.{info.name}"
        try:
            child = importlib.import_module(cname)
            yield child
            if recursive:
                yield from children(child)
        except ImportError:
            logging.warning(f"[warning] import error at {cname}")


def gen_rst(module, f):
    name = module.__name__
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

    cs = children(module, recursive=True)
    for c in cs:
        cname = c.__name__
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
for c in children(root, recursive=False):
    fname = c.__name__.replace(".", "-") + ".rst"
    dst = f"{gendir}/{fname}"
    modules_rst += f"   ./_gen/{fname}\n"
    with open(dst, "w") as f:
        gen_rst(c, f)


with open(gendir + "/modules.rst", "w") as f:
    f.write(modules_rst)
