"""Tests of the licence ids of local/v2/manifest_common.py.

python3 -m pytest local/v2/test_manifest_common.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from manifest_common import licence_accepted, licence_id  # noqa: E402


@pytest.mark.parametrize(
    "text, want",
    [
        ("Creative Commons License: Attribution 4.0 International", "CC-BY-4.0"),
        (
            "This work is licensed under the Creative Commons Attribution 4.0 "
            "International License. ... Mountain View, CA 94042, USA.",
            "CC-BY-4.0",
        ),
        ("cc-by-4.0 https://creativecommons.org/licenses/by/4.0/deed.de", "CC-BY-4.0"),
        ("CC BY 3.0 AT", "CC-BY-3.0-AT"),
        ("CC BY", "CC-BY"),
        ("CC0/Public Domain", "CC0-1.0"),
        ("http://creativecommons.org/licenses/by/3.0/de/", "CC-BY-3.0-DE"),
        ("https://creativecommons.org/licenses/by/4.0/de/", "CC-BY-4.0"),
        ("http://creativecommons.org/licenses/by-sa/3.0/", "CC-BY-SA-3.0"),
        ("http://creativecommons.org/licenses/by-nc-nd/3.0/", "CC-BY-NC-ND-3.0"),
        ("http://creativecommons.org/publicdomain/mark/1.0/", "PDM-1.0"),
        ("CC BY-SA 4.0", "CC-BY-SA-4.0"),
        ("© 2026 Luis und Steffen", ""),
    ],
)
def test_licence_id(text, want):
    assert licence_id(text) == want


@pytest.mark.parametrize(
    "lid, ok",
    [
        ("CC-BY-4.0", True),
        ("CC-BY-3.0-AT", True),
        ("CC-BY", True),
        ("CC0-1.0", True),
        ("PDM-1.0", True),
        ("CC-BY-SA-4.0", False),
        ("CC-BY-NC-4.0", False),
        ("CC-BY-ND-4.0", False),
        ("", False),
    ],
)
def test_licence_accepted(lid, ok):
    assert licence_accepted(lid) is ok
