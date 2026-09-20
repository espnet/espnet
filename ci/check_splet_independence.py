#!/usr/bin/env python3
"""Check that splet/ stays separable from the rest of the repository.

SPLET lives in this repository for now and is meant to be split out into its
own repository and PyPI distribution later (espnet/espnet#6760). That is only
true for as long as nothing inside splet/ reaches into ESPnet. One
`from espnet2.text.cleaner import TextCleaner` -- which is exactly what the
espnet3 metrics it will replace do today -- and the extraction stops being a
directory move and becomes a porting project.

The rule is one-directional and deliberate: espnet3 may call splet, splet may
not call espnet. So this check fails on:

1. Any import of espnet, espnet2, espnet3, torch, lightning or any other
   training-stack package from inside splet/. A text evaluator has no use for
   a tensor library, and pulling one in would make the extracted package
   inherit ESPnet's install.

2. Any import of a third-party package that is not declared. Declared means
   the `splet` extra in pyproject.toml, plus the packages splet imports
   behind a try/except with a message telling the user what to install.
   An undeclared import works on a developer's machine, where everything is
   installed, and fails for the first person who installs only what the
   package asks for.

It does not check the direction that is allowed: espnet3 importing splet is
the point.
"""

import ast
import pathlib
import sys
import tomllib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PACKAGE = ROOT / "splet"

# Imports that make the package inseparable from ESPnet, with the reason
# each one is refused rather than a bare list.
FORBIDDEN = {
    "espnet": "SPLET must not depend on ESPnet; the dependency runs the other way",
    "espnet2": "SPLET must not depend on ESPnet; the dependency runs the other way",
    "espnet3": "SPLET must not depend on ESPnet; the dependency runs the other way",
    "torch": "a text evaluator does not need a tensor library",
    "torchaudio": "a text evaluator does not need an audio library",
    "lightning": "a text evaluator does not need a training framework",
    "pytorch_lightning": "a text evaluator does not need a training framework",
    "hydra": "SPLET is configured by a plain YAML list, as VERSA is",
    "omegaconf": "SPLET is configured by a plain YAML list, as VERSA is",
    "espnet_model_zoo": "SPLET scores text it is given; it does not load models",
}

# Third-party packages splet may import. Anything here is either in the
# `splet` extra of pyproject.toml or imported behind a try/except that names
# the package to install.
ALLOWED_THIRD_PARTY = {
    "yaml",  # required: the score config
    "rapidfuzz",  # optional: linear-memory alignment
    "scipy",  # optional: the assignment step of speaker-permuted metrics
    "sacrebleu",  # optional: MT metrics, once the corpus tier has them
    "sentencepiece",  # optional: token error rate
}


def imported_modules(path):
    """Yield (top-level module, line number) for every import in a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield alias.name.split(".")[0], node.lineno
        elif isinstance(node, ast.ImportFrom):
            # A relative import stays inside the package by construction.
            if node.level == 0 and node.module:
                yield node.module.split(".")[0], node.lineno


def main():
    """Report every import that would make splet/ inseparable."""
    if not PACKAGE.is_dir():
        print(f"{PACKAGE} does not exist", file=sys.stderr)
        return 1

    declared = set(ALLOWED_THIRD_PARTY)
    failures = []
    for path in sorted(PACKAGE.rglob("*.py")):
        for module, lineno in imported_modules(path):
            where = f"{path.relative_to(ROOT)}:{lineno}"
            if module in FORBIDDEN:
                failures.append(f"{where}: imports {module} -- {FORBIDDEN[module]}")
            elif (
                module not in sys.stdlib_module_names
                and module not in declared
                and module != "splet"
            ):
                failures.append(
                    f"{where}: imports undeclared third-party package "
                    f"'{module}'. Add it to the `splet` extra in "
                    f"pyproject.toml and to ALLOWED_THIRD_PARTY here, or "
                    f"import it behind a try/except."
                )

    if failures:
        print("splet/ is no longer separable from the rest of the repo:\n")
        for failure in failures:
            print(f"  {failure}")
        print(
            "\nSPLET is meant to be extracted into its own repository "
            "(espnet/espnet#6760). espnet3 may import splet; splet may not "
            "import espnet."
        )
        return 1

    extra = tomllib.loads((ROOT / "pyproject.toml").read_text())
    extra_names = extra["project"]["optional-dependencies"].get("splet")
    if extra_names is None:
        print("pyproject.toml declares no `splet` extra", file=sys.stderr)
        return 1

    print(
        f"splet/ imports nothing from ESPnet and nothing undeclared "
        f"({len(list(PACKAGE.rglob('*.py')))} files checked); "
        f"the `splet` extra declares {len(extra_names)} package(s)."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
