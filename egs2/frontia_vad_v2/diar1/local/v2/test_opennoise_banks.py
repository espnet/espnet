"""Tests of local/v2/opennoise_banks.py on synthetic OpenNoise lists."""

import csv
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent / "opennoise_banks.py"
COLS = [
    "entry",
    "start_s",
    "end_s",
    "kind",
    "source",
    "split",
    "duration_s",
    "spdx",
    "licence_class",
    "speech_flag",
    "vocal_flag",
    "speech_check",
]


def lists(tmp_path, pieces, rirs=("r1",)):
    d = tmp_path / "lists"
    d.mkdir(exist_ok=True)
    for kind in ("noise", "music", "babble"):
        ps = [p for p in pieces if p["kind"] == kind]
        entries = sorted({p["entry"] for p in ps})
        (d / f"{kind}_train.scp").write_text(
            "".join(f"{e} /audio/{e}.wav\n" for e in entries)
        )
        with open(d / f"{kind}_train_manifest.tsv", "w", newline="") as f:
            w = csv.DictWriter(f, COLS, delimiter="\t")
            w.writeheader()
            w.writerows(ps)
    (d / "rir_train.scp").write_text("".join(f"{r} /audio/{r}.wav\n" for r in rirs))
    return d


def piece(
    entry,
    kind,
    start,
    src="s",
    spdx="CC0-1.0",
    cls="PD/CC0",
    sf="0",
    vf="0",
    check="keep ratio=0.00",
):
    return dict(
        entry=entry,
        start_s=start,
        end_s=start + 30,
        kind=kind,
        source=src,
        split="train",
        duration_s=30,
        spdx=spdx,
        licence_class=cls,
        speech_flag=sf,
        vocal_flag=vf,
        speech_check=check,
    )


def run(tmp_path, pieces, *extra):
    d = lists(tmp_path, pieces)
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--lists",
            str(d),
            "--out",
            str(tmp_path / "out"),
            *extra,
        ],
        capture_output=True,
        text=True,
    )


def rows(path):
    return list(csv.DictReader(open(path), delimiter="\t"))


PIECES = [
    piece("noise-a-0", "noise", 0),
    piece("noise-a-0", "noise", 30, spdx="CC-BY-4.0", cls="BY"),
    piece("noise-b-0", "noise", 0, src="b", check="noise ratio=0.03"),  # babble-routed
    piece("music-m-0", "music", 0, src="m"),
    piece("babble-c-0", "babble", 0, src="c", sf="1", check="babble ratio=0.6 words=1"),
]


def test_pools_packs_licences_and_flags(tmp_path):
    r = run(tmp_path, PIECES)
    assert r.returncode == 0, r.stderr
    out = tmp_path / "out"
    noise = rows(out / "noise_pool.tsv")
    assert [x["id"] for x in noise] == ["noise-a-0"]  # the babble-routed pack left out
    assert (
        noise[0]["licence"] == "CC-BY-4.0+CC0-1.0"
        and float(noise[0]["duration_s"]) == 60
    )
    assert noise[0]["flagged"] == "0"
    both = rows(out / "noise_babble_pool.tsv")
    assert [(x["id"], x["flagged"]) for x in both] == [
        ("noise-a-0", "0"),
        ("babble-c-0", "1"),
    ]
    assert [x["id"] for x in rows(out / "music_pool.tsv")] == ["music-m-0"]
    assert (out / "rir.scp").read_text().split()[0] == "r1"


def test_babble_routed_noise_kept_on_request(tmp_path):
    r = run(tmp_path, PIECES, "--keep_babble_routed_noise")
    assert r.returncode == 0, r.stderr
    assert len(rows(tmp_path / "out" / "noise_pool.tsv")) == 2


def test_a_flagged_noise_or_music_piece_stops(tmp_path):
    for p in (
        piece("music-v-0", "music", 0, vf="1"),
        piece("noise-v-0", "noise", 0, sf="1"),
    ):
        r = run(tmp_path, PIECES + [p])
        assert r.returncode != 0 and "flagged" in r.stderr


def test_a_licence_class_outside_the_list_stops(tmp_path):
    r = run(
        tmp_path,
        PIECES + [piece("noise-x-0", "noise", 0, spdx="CC-BY-SA-4.0", cls="SA")],
    )
    assert r.returncode != 0 and "licence class" in r.stderr
