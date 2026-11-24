"""Tests of attribution_manifest.py that need no corpus."""

import gzip
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import attribution_manifest as am  # noqa: E402

HEADER = (
    "entry\tstart_s\tend_s\tsegment\tparent\tkind\tsource\tsplit\tspdx\t"
    "licence_class\tlicence\tauthor\ttitle\torig_url\torig_id\tmodified\t"
    "duration_s\tspeech_flag\tspeech_reason\tvocal_flag\tspeech_check\n"
)


def test_freesound_url_and_id():
    assert am.freesound_url("345111") == "https://freesound.org/s/345111/"
    assert am.freesound_url("abc") == ""
    assert (
        am.freesound_id_of("https://freesound.org/apiv2/sounds/211145/download/")
        == "211145"
    )
    assert am.freesound_id_of("https://freesound.org/s/42/") == "42"
    assert am.freesound_id_of("https://example.org/") == ""


def test_clean_makes_one_cell():
    assert am.clean("a\tb\nc  d") == "a b c d"


def test_augmentation_rows_keep_only_the_used_packs(tmp_path):
    m = tmp_path / "noise_train_manifest.tsv"
    m.write_text(
        HEADER
        + "noise-x-00000\t0.000\t2.500\tx-1\tx-1\tnoise\tx\ttrain\tCC-BY-4.0\tBY\t"
        "cc-by\tAlice\tRain\thttps://freesound.org/s/1/\t1\t\t2.5\t0\t\t0\t\n"
        + "noise-x-00001\t0.000\t1.000\tx-2\tx-2\tnoise\tx\ttrain\tCC0-1.0\tPD/CC0\t"
        "cc0\tBob\tWind\thttps://freesound.org/s/2/\t2\t\t1.0\t0\t\t0\t\n"
    )
    rows = list(am.augmentation_rows("noise", {"noise-x-00000"}, m))
    assert rows == [
        (
            "noise",
            "x",
            "noise-x-00000",
            "x-1",
            "CC-BY-4.0",
            "Alice",
            "Rain",
            "https://freesound.org/s/1/",
            2.5,
        )
    ]


def test_main_counts_rows_and_uncredited(tmp_path, monkeypatch, capsys):
    """Augmentation-only run: speech_rows is replaced by a fixed record."""
    lists = tmp_path / "lists"
    lists.mkdir()
    for kind in ("noise", "music", "rir"):
        (lists / f"{kind}_train_manifest.tsv").write_text(
            HEADER
            + f"{kind}-y-0\t0.000\t3.000\ty-1\ty-1\t{kind}\ty\ttrain\tCC-BY-4.0\tBY\t"
            "cc-by\t\tT\thttps://example.org/1\t1\t\t3.0\t0\t\t0\t\n"
        )
    for kind in ("noise", "music"):
        (tmp_path / f"{kind}_pool.tsv").write_text(
            "id\tpath\tkind\tsource\tlicence\tduration_s\tflagged\n"
            f"{kind}-y-0\tp\t{kind}\ty\tCC-BY-4.0\t3.0\t0\n"
        )
    (tmp_path / "rir.scp").write_text("rir-y-0 /x.wav\n")
    dump = tmp_path / "dump"
    dump.mkdir()
    monkeypatch.setattr(
        am,
        "speech_rows",
        lambda cr, manifest: iter(
            [("speech", "c", "r1", "a.wav", "CC0-1.0", "", "", "", 10.0)]
        ),
    )
    out = tmp_path / "m.tsv.gz"
    am.main(
        [
            str(tmp_path),
            str(dump),
            str(tmp_path / "noise_pool.tsv"),
            str(tmp_path / "music_pool.tsv"),
            str(tmp_path / "rir.scp"),
            str(lists),
            str(out),
        ]
    )
    lines = gzip.open(out, "rt").read().splitlines()
    assert lines[0].split("\t") == list(am.COLUMNS)
    assert len(lines) == 5
    printed = capsys.readouterr().out
    assert "speech: 1 records, 1 rows" in printed
    # the three BY rows have no author
    assert printed.count(" y ") + printed.count("y      ") >= 1
    for kind in ("noise", "music", "rir"):
        assert f"{kind}: 1 records, 1 rows" in printed


def test_licence_refs_are_classified():
    assert am.licence_class("LicenseRef-public-domain") == "PD/CC0"
    assert am.licence_class("LicenseRef-UIowa-MIS-free-use") == "permissive"
    assert am.licence_class("generated") == "generated"
    assert am.licence_class("CC-BY-4.0") == "BY"
