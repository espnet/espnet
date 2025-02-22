#!/usr/bin/env python3
from glob import glob
import importlib  # noqa
import os
import ast
import sys  # noqa
import subprocess

import configargparse


def get_git_revision_hash() -> str:
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


GIT_HASH = get_git_revision_hash()


def to_module(path_name):
    ret = path_name.replace(".py", "").replace("/", ".")
    if ret.endswith("."):
        return ret[:-1]
    return ret


def top_level_functions(body):
    return (
        f for f in body
        if isinstance(f, ast.FunctionDef)
        and not f.name.startswith("_")
    )


def top_level_classes(body):
    return (f for f in body if isinstance(f, ast.ClassDef))


def parse_ast(filename):
    with open(filename, "rt") as file:
        return ast.parse(file.read(), filename=filename)


def gen_func_rst(func_name, writer, filepath, lineno):
    sourceurl = "https://github.com/espnet/espnet/blob/" \
        + GIT_HASH + "/" + filepath + f"#L{lineno}"
    writer.write(f""".. _{func_name}
{func_name}
{"~" * len(func_name)}

`source <{sourceurl}>`_

.. autofunction:: {func_name}
""")


def gen_class_rst(class_name, writer, filepath, lineno):
    sourceurl = "https://github.com/espnet/espnet/blob/" \
        + GIT_HASH + "/" + filepath + f"#L{lineno}"
    writer.write(f""".. _{class_name}
{class_name}
{"~" * len(class_name)}

`source <{sourceurl}>`_

.. autoclass:: {class_name}
    :members:
    :undoc-members:
    :show-inheritance:
""")


if __name__ == "__main__":
    # parser
    parser = configargparse.ArgumentParser(
        description="generate RST files from <root> module recursively into <dst>/_gen",
        config_file_parser_class=configargparse.YAMLConfigFileParser,
        formatter_class=configargparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--root", type=str, help="root module to generate docs"
    )
    parser.add_argument("--dst", type=str, help="destination path to generate RSTs")
    parser.add_argument("--exclude", nargs="*", default=[], help="exclude module name")
    args = parser.parse_args()
    print(args)

    gendir = args.dst
    os.makedirs(gendir, exist_ok=True)
    os.makedirs(f"{gendir}/{args.root}", exist_ok=True)

    for p in glob(args.root + "/**", recursive=True):
        module_name = to_module(p)
        if any([ex in module_name for ex in args.exclude]):
            continue
        if "__init__" in p:
            continue
        if not p.endswith(".py"):
            continue

        submodule_name = module_name.split(".")[1]
        os.makedirs(f"{gendir}/{args.root}/{submodule_name}", exist_ok=True)

        if not os.path.exists(f"{gendir}/{args.root}/{submodule_name}/README.rst"):
            # 1 get functions
            for func in top_level_functions(parse_ast(p).body):
                function_name = func.name
                print(f"[INFO] generating {func.name} in {module_name}")
                # 1.2 generate RST
                with open(
                    f"{gendir}/{args.root}/{submodule_name}/{function_name}.rst", "w"
                ) as f_rst:
                    gen_func_rst(
                        f"{module_name}.{function_name}",
                        f_rst,
                        p,
                        func.lineno
                    )

            # 2 get classes
            for clz in top_level_classes(parse_ast(p).body):
                class_name = clz.name
                print(f"[INFO] generating {clz.name} in {module_name}")
                # 1.2 generate RST
                with open(
                    f"{gendir}/{args.root}/{submodule_name}/{class_name}.rst", "w"
                ) as f_rst:
                    gen_class_rst(f"{module_name}.{class_name}", f_rst, p, clz.lineno)
