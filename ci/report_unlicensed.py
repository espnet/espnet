#!/usr/bin/env python3
"""Report installed distributions that declare no licence.

Run during the prebuilt CI image build. This does not fail the build: licence
metadata is inconsistent enough that failing on it would block for bad reasons.
It exists so that a new dependency arriving without a licence is *visible*, and
someone can decide whether it belongs in ci/no_redistribute.txt.

kaldiio is the reason this exists. Its licence forbids redistribution, and
nothing about installing it made that apparent.
"""

import sys
from importlib.metadata import distributions
from pathlib import Path


def declared_licence(meta) -> str:
    for key in ("License-Expression", "License"):
        value = meta.get(key)
        if value and value.strip() and value.strip().upper() not in {"UNKNOWN", "NONE"}:
            return value.strip().splitlines()[0][:60]
    for classifier in meta.get_all("Classifier") or ():
        if classifier.startswith("License ::"):
            return classifier.split("::")[-1].strip()[:60]
    return ""


def already_listed(root: Path) -> set:
    path = root / "ci" / "no_redistribute.txt"
    if not path.is_file():
        return set()
    names = set()
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if line:
            for sep in "<>=!~;[":
                line = line.split(sep)[0]
            names.add(line.strip().lower().replace("_", "-"))
    return names


def main() -> int:
    listed = already_listed(Path(__file__).resolve().parent.parent)
    unlicensed = []
    for dist in distributions():
        name = (dist.metadata["Name"] or "").strip()
        if not name:
            continue
        if name.lower().replace("_", "-") in listed:
            continue
        if not declared_licence(dist.metadata):
            unlicensed.append(name)

    print(f"::group::Distributions with no declared licence ({len(set(unlicensed))})")
    for name in sorted(set(unlicensed), key=str.lower):
        print(f"  {name}")
    print("::endgroup::")
    if unlicensed:
        print(
            "Note: no licence metadata is not the same as no licence, but each of "
            "these is worth a look. If one turns out to forbid redistribution, add "
            "it to ci/no_redistribute.txt so it stays out of the image."
        )
    if listed:
        print(f"Already excluded from the image: {', '.join(sorted(listed))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
