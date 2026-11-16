"""Tests of the per-file licence filter (adapters_nonspeech.licence_ok, MUSAN LICENSE).

python3 -m pytest local/v2/test_licences.py
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from adapters_nonspeech import (  # noqa: E402
    _musan_licences,
    licence_ok,
    musan_licence_table,
)


@pytest.mark.parametrize(
    "text",
    [
        "CC BY 3.0",
        "Attribution 3.0 Unported (CC BY 3.0)",
        "License: Attribution 3.0",
        "Creative Commons Attribution License.",
        "License:Public Domain",
        "All selected recordings were marked as in the Public Domain",
        "http://creativecommons.org/publicdomain/zero/1.0/",
        "https://creativecommons.org/licenses/by/4.0/",
    ],
)
def test_permissive(text):
    assert licence_ok(text)


@pytest.mark.parametrize(
    "text",
    [
        "CC BY-SA 4.0",
        "License:CC BY-SA 3.0 DE",
        "Attribution-ShareAlike 3.0 Unported (CC BY-SA 3.0)",
        "is licensed under a Attribution-Share Alike 3.0 United States License.",
        "Attribution Share-Alike",
        "Attribution-NoDerivs 3.0 Unported (CC BY-ND 3.0)",
        "Attribution No Derivatives",
        "Attribution-NonCommercial 4.0",
        "Attribution Non Commercial",
        "http://creativecommons.org/licenses/by-nc/3.0/",
        "https://creativecommons.org/licenses/by-sa/3.0/",
        "https://creativecommons.org/licenses/sampling+/1.0/",
        "",
        "All rights reserved",
    ],
)
def test_not_permissive(text):
    assert not licence_ok(text)


LICENSE = """CC BY 3.0
http://example.org/whole-directory
==============================================================================
music-x-0001
"Title one" (by A)
music-x-0002
"Title two" (by B)
==============================================================================
music-x-0003
"Title three" (by C)
Selections by D is licensed under a Attribution-Share Alike 3.0 United States License.
http://example.org/d
==============================================================================
music-x-0004
Attribution-NoDerivs 3.0 Unported (CC BY-ND 3.0)
==============================================================================
music-x-0005
License:Public Domain
http://example.org/pd
"""


def test_musan_blocks(tmp_path):
    lic = tmp_path / "LICENSE"
    lic.write_text(LICENSE)
    stems = [
        "music-x-0001",
        "music-x-0002",
        "music-x-0003",
        "music-x-0004",
        "music-x-0005",
        "music-x-0099",
    ]
    table = musan_licence_table(lic, stems)
    # an id block without a licence line takes the directory's licence and URL
    assert table["music-x-0001"] == ("CC BY 3.0", "http://example.org/whole-directory")
    assert table["music-x-0003"][1] == "http://example.org/d"
    assert table["music-x-0099"] == ("", "")  # in no block: no licence
    ok = _musan_licences(lic, stems)
    assert ok == {
        "music-x-0001": True,
        "music-x-0002": True,
        "music-x-0003": False,  # "Share Alike" with a space
        "music-x-0004": False,
        "music-x-0005": True,
        "music-x-0099": False,
    }


def test_musan_whole_directory_licence(tmp_path):
    lic = tmp_path / "LICENSE"
    lic.write_text(
        "All selected recordings were marked as in the Public Domain\n"
        "on Free Sound (url: https://www.freesound.org/).\n"
    )
    ok = _musan_licences(lic, ["noise-free-sound-0001"])
    assert ok == {"noise-free-sound-0001": True}
    assert musan_licence_table(lic, ["a"])["a"][1].startswith(
        "https://www.freesound.org"
    )
    assert _musan_licences(tmp_path / "missing", ["a"]) == {"a": False}


# Per-file licences of the real MUSAN LICENSE files (2016 release), counted file by
# file: (CC BY or PD files, files with SA, ND or no licence). Set MUSAN_ROOT to the
# directory holding music/ and noise/ to run it.
MUSAN_EXPECTED = {
    "music/fma": (85, 43),  # 78 BY + 7 PD; SA 37, ND 2, none 4
    "music/fma-western-art": (58, 35),  # 29 BY + 29 PD; SA 26 ("Share Alike"), none 9
    "music/hd-classical": (19, 56),  # 1 BY + 18 PD; SA 56
    "music/jamendo": (27, 190),  # 18 BY + 9 PD; SA 156, ND 28, none 6
    "music/rfm": (147, 0),  # CC BY 3.0 for the whole directory
    "noise/free-sound": (843, 0),  # public domain
    "noise/sound-bible": (87, 0),  # 67 BY + 20 PD
}


@pytest.mark.skipif(
    not __import__("os").environ.get("MUSAN_ROOT"), reason="set MUSAN_ROOT"
)
@pytest.mark.parametrize("sub", sorted(MUSAN_EXPECTED))
def test_real_musan_licences(sub):
    import os

    d = Path(os.environ["MUSAN_ROOT"]) / sub
    stems = [p.stem for p in d.glob("*.wav")]
    ok = _musan_licences(d / "LICENSE", stems)
    kept = sum(ok.values())
    assert (kept, len(stems) - kept) == MUSAN_EXPECTED[sub]
