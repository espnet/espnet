#!/usr/bin/env python3
"""The support matrix in README.md, checked against the code it describes.

A table of what runs where is exactly the kind of thing that is true when it
is written and quietly false a month later: a command is renamed, a tool is
added, and the table still says what someone remembered. So the table is
derived here from the code and compared, rather than trusted.

    python ci/check_front_ends.py            # print the matrix, check the table

What is read:

  espnet2/bin/cli.py         the subcommands, from the add(...) calls
  espnet2/bin/mcp_server.py  the tools, from what build_server registers
  README.md                  the table, whose CLI and MCP columns must agree

Spaces and notebooks are links rather than code, and `ci/check_demo_links.py`
already fetches every one of them daily; this does not repeat that.

doc/front_ends.md is the prose this enforces a corner of.
"""

import ast
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
CLI = ROOT / "espnet2" / "bin" / "cli.py"
MCP = ROOT / "espnet2" / "bin" / "mcp_server.py"
README = ROOT / "README.md"

# Subcommands that are not a task: they run no model of their own, and the
# matrix is about tasks.
NOT_A_TASK = {"demo", "models"}


def cli_commands() -> set:
    """The verbs `espnet` offers, from the add(...) calls in build_parser."""
    found = set()
    for node in ast.walk(ast.parse(CLI.read_text())):
        if not isinstance(node, ast.Call):
            continue
        named = getattr(node.func, "id", None)
        if named == "add" and node.args and isinstance(node.args[0], ast.Constant):
            found.add(node.args[0].value)
        # `espnet models` is added straight on the subparsers
        if (
            isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_parser"
            and node.args
            and isinstance(node.args[0], ast.Constant)
        ):
            found.add(node.args[0].value)
    return found - NOT_A_TASK


def mcp_tools() -> set:
    """The tools the MCP server registers, from the TOOLS it iterates.

    Read from the module rather than by importing it: importing wants the
    `mcp` package, and what is being checked is the source anyway.
    """
    tree = ast.parse(MCP.read_text())
    defined = {
        f.name
        for f in tree.body
        if isinstance(f, ast.FunctionDef) and not f.name.startswith("_")
    }
    for node in tree.body:
        if not isinstance(node, ast.Assign):
            continue
        if not any(
            isinstance(target, ast.Name) and target.id == "TOOLS"
            for target in node.targets
        ):
            continue
        listed = {
            element.id
            for element in ast.walk(node.value)
            if isinstance(element, ast.Name)
        }
        return listed & defined
    raise SystemExit("mcp_server.py defines no TOOLS")


ROW = re.compile(r"^\|\s*\*\*(?P<task>[A-Z]+)\*\*[^|]*\|(?P<cells>.*)\|\s*$", re.M)
CODE = re.compile(r"`([^`]+)`")


def matrix_rows():
    """The table's rows, as {task: [space, notebook, mcp, cli]} of cell text."""
    rows = {}
    for match in ROW.finditer(README.read_text()):
        cells = [c.strip() for c in match.group("cells").split("|")]
        if len(cells) != 4:
            continue
        rows[match.group("task")] = cells
    return rows


def main() -> int:
    commands, tools = cli_commands(), mcp_tools()
    rows = matrix_rows()
    if not rows:
        print("README.md has no support matrix to check", file=sys.stderr)
        return 1

    problems = []
    claimed_cli, claimed_mcp = set(), set()
    for task, (space, notebook, mcp, cli) in sorted(rows.items()):
        print(f"{task:6} space={space[:18]:18} notebook={notebook[:18]:18} ")
        for cell, kind, offered, claimed in (
            (cli, "CLI", commands, claimed_cli),
            (mcp, "MCP", tools, claimed_mcp),
        ):
            named = [n.removeprefix("espnet ") for n in CODE.findall(cell)]
            for name in named:
                claimed.add(name)
                if name not in offered:
                    problems.append(
                        f"{task}: the table says {kind} has `{name}`, "
                        f"which {'cli.py' if kind == 'CLI' else 'mcp_server.py'} "
                        f"does not offer"
                    )
            if named and "❌" in cell:
                problems.append(f"{task}: the {kind} cell names {named} and says ❌")
            if not named and "🟢" in cell:
                problems.append(f"{task}: the {kind} cell is 🟢 but names nothing")

    for name in sorted(commands - claimed_cli):
        problems.append(f"`espnet {name}` is a command the table does not list")
    for name in sorted(tools - claimed_mcp):
        problems.append(f"`{name}` is an MCP tool the table does not list")

    print(f"\n{len(commands)} commands, {len(tools)} MCP tools, {len(rows)} rows")
    for problem in problems:
        print(f"  {problem}", file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
