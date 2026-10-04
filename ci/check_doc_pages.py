#!/usr/bin/env python3
"""Two documentation pages must not arrive at the same address.

`vuepress-plugin-search-pro` indexes a page by its path: `addItem(page.path)`
returns the index of that path in a list, so two pages with one path are
handed the same id and the build dies with

    Error: SlimSearch: duplicate ID 1132

twenty-five minutes in, naming neither page. That happened when
`espnet2/legacy/_Hypothesis` joined `espnet2/legacy/Hypothesis`: a leading
underscore is not part of the address the site serves, so both became
`/guide/espnet2/legacy/Hypothesis.html`.

This says which two pages, before the build starts.

    python ci/check_doc_pages.py [root]      # default: the generated rst

The check is on the generated `.rst`, which is where a collision is created,
rather than on the built markdown - so it runs in a second and needs neither
sphinx nor node.
"""

import collections
import pathlib
import sys

DEFAULT_ROOT = pathlib.Path(__file__).resolve().parent.parent / "doc" / "_gen"


def address(path: pathlib.Path) -> str:
    """What the site will serve this page as.

    VuePress runs every path segment through `sanitizeFileName`, which strips
    a leading underscore - checked against @vuepress/utils 2.0.0-rc.14:

        _Hypothesis.html -> Hypothesis.html
        _RNNTNumba.html  -> RNNTNumba.html
        Stft.html        -> Stft.html

    Case is left alone there, so it is left alone here: `Stft` and `stft` are
    two pages on a case-sensitive filesystem and the site serves both.
    """
    return "/".join(part.lstrip("_") for part in path.with_suffix("").parts)


def main(argv) -> int:
    root = pathlib.Path(argv[1]) if len(argv) > 1 else DEFAULT_ROOT
    if not root.is_dir():
        print(f"{root} does not exist; nothing to check", file=sys.stderr)
        return 0

    seen = collections.defaultdict(list)
    for page in sorted(root.rglob("*.rst")):
        seen[address(page.relative_to(root))].append(page.relative_to(root))

    clashes = {k: v for k, v in seen.items() if len(v) > 1}
    print(f"{len(seen)} pages under {root}")
    for where, pages in sorted(clashes.items()):
        print(
            f"  {where}: " + ", ".join(str(p) for p in pages),
            file=sys.stderr,
        )
    if clashes:
        print(
            f"{len(clashes)} address(es) claimed by more than one page. The "
            "search index keys on the address, so the build fails with a "
            "duplicate id and no page name. Rename one, or stop generating "
            "it.",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
