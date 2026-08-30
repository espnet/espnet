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


def _names(path: Path) -> set:
    """First whitespace-separated field of each non-comment line, normalised."""
    if not path.is_file():
        return set()
    names = set()
    for line in path.read_text().splitlines():
        line = line.split("#")[0].strip()
        if not line:
            continue
        field = line.split()[0]
        for sep in "<>=!~;[":
            field = field.split(sep)[0]
        names.add(field.strip().lower().replace("_", "-"))
    return names


def excluded(root: Path) -> set:
    """Packages kept out of the image because they forbid redistribution."""
    return _names(root / "ci" / "no_redistribute.txt")


def audited(root: Path) -> set:
    """Packages with no licence metadata that were checked and are fine."""
    return _names(root / "ci" / "licence_audited.txt")


def main() -> int:
    root = Path(__file__).resolve().parent.parent
    listed = excluded(root)
    known = audited(root)
    skip = listed | known
    unlicensed = []
    for dist in distributions():
        name = (dist.metadata["Name"] or "").strip()
        if not name:
            continue
        if name.lower().replace("_", "-") in skip:
            continue
        if not declared_licence(dist.metadata):
            unlicensed.append(name)

    fresh = sorted(set(unlicensed), key=str.lower)
    print(
        f"::group::Distributions with no declared licence, not yet checked ({len(fresh)})"
    )
    for name in fresh:
        print(f"  {name}")
    print("::endgroup::")
    if fresh:
        print(
            "Note: no licence metadata is not the same as no licence, but each of "
            "these is worth a look. Record the outcome in ci/licence_audited.txt, "
            "or in ci/no_redistribute.txt if it turns out to forbid redistribution."
        )
    if listed:
        print(f"Excluded from the image: {', '.join(sorted(listed))}")
    if known:
        print(f"Checked previously, redistribution permitted: {len(known)} packages")
    return 0


if __name__ == "__main__":
    sys.exit(main())
