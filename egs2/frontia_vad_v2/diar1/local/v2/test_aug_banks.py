"""Tests of local/v2/aug_banks.py: packing, the train-split filter and licences."""

import sys
from pathlib import Path

import numpy as np
import soundfile

sys.path.insert(0, str(Path(__file__).resolve().parent))

from aug_banks import FS, Collector, PackWriter  # noqa: E402
from registry import AUTO  # noqa: E402


def test_packs_concatenate_clips_and_record_their_spans(tmp_path):
    pw = PackWriter(tmp_path, "music", "src", pack_seconds=1.0)
    for i, n in enumerate([6000, 6000, 6000, 3000]):
        pw.add(f"c{i}", np.full(n, 0.1 * (i + 1), np.float32), "CC0", f"m{i}")
    pw.flush()
    assert [p.name for p, _ in pw.files] == ["src_00000.wav", "src_00001.wav"]
    # a pack is closed after the clip that reaches pack_seconds
    x, rate = soundfile.read(pw.files[0][0])
    assert rate == FS and len(x) == 18000
    spans = [(c, s, e) for c, s, e, _, _ in pw.files[0][1]]
    assert spans == [("c0", 0.0, 0.375), ("c1", 0.375, 0.75), ("c2", 0.75, 1.125)]
    # the clip at 0.375 s is the second one
    assert abs(x[6000] - 0.2) < 1e-3
    assert [c for c, *_ in pw.files[1][1]] == ["c3"]
    assert abs(pw.hours() - 21000 / FS / 3600) < 1e-12


def test_collector_keeps_train_clips_resampled_first_channel(tmp_path):
    pw = PackWriter(tmp_path, "effects", "src", pack_seconds=1000.0)
    col = Collector(pw, lambda r: f"lic-{r}")
    stereo = np.stack([np.ones(8000), -np.ones(8000)], 1).astype(np.float32)
    assert col["train"].add("a", stereo, 8000, channels=[0], source="m/a") == 1
    assert col["dev"].add("b", stereo, 8000) == 0
    # a split by group follows registry.hash_split, as prep.py does
    kept = sum(col[f"{AUTO}g{i}"].add(f"g{i}", stereo, 8000) for i in range(200))
    assert 180 < kept < 200
    pw.flush()
    ((path, clips),) = pw.files
    x, _ = soundfile.read(path)
    assert len(x) == 16000 * (1 + kept)  # 1 s each at 16 kHz
    # channel 0 (+1, not -1); the pack is scaled down by the resampler's overshoot
    assert x[100:15900].min() > 0.8
    assert clips[0][3:] == ["lic-a", "m/a"]
    assert col.skipped == 1 + 200 - kept


def test_collector_stops_at_max_hours(tmp_path):
    pw = PackWriter(tmp_path, "effects", "src", pack_seconds=10.0)
    col = Collector(pw, lambda r: "", max_hours=2.5 / 3600)
    added = [col["train"].add(f"c{i}", np.ones(FS, np.float32), FS) for i in range(5)]
    assert added == [1, 1, 1, 0, 0]


def test_write_lists_filters_per_file_and_records_urls(tmp_path):
    import csv
    from types import SimpleNamespace

    from aug_banks import read_packs, write_lists

    root = tmp_path / "corpora"
    music = root / "musan" / "music" / "x"
    noise = root / "musan" / "noise" / "y"
    for d in (music, noise):
        d.mkdir(parents=True)
    for stem in ("music-x-0001", "music-x-0002"):
        soundfile.write(music / f"{stem}.wav", np.zeros(160, np.float32), FS)
    (music / "LICENSE").write_text(
        "music-x-0001\nCC BY 3.0\nhttp://a\n=====\n"
        "music-x-0002\nAttribution-Share Alike 3.0 United States License.\nhttp://b\n"
    )
    (music / "ANNOTATIONS").write_text("music-x-0001.wav g N\nmusic-x-0002.wav g N\n")
    soundfile.write(noise / "noise-y-0001.wav", np.zeros(160, np.float32), FS)
    (noise / "LICENSE").write_text("Public Domain\nhttp://pd\n")
    (root / "chime3_background").mkdir()
    out = tmp_path / "banks"
    out.mkdir()
    good, bad = out / "music" / "fsd_00000.wav", out / "music" / "fsd_00001.wav"
    packs = [
        ("music", "fsd50k_music", good, [["c1", 0.0, 1.0, "http://cc0", "m1"]]),
        (
            "music",
            "fsd50k_music",
            bad,
            [
                ["c2", 0.0, 1.0, "https://creativecommons.org/licenses/by/3.0/", "m2"],
                [
                    "c3",
                    1.0,
                    2.0,
                    "https://creativecommons.org/licenses/by-sa/3.0/",
                    "m3",
                ],
            ],
        ),
        (
            "music",
            "slakh2100",
            out / "music" / "slakh_00000.wav",
            [["t", 0, 9, "", "s"]],
        ),
    ]
    a = SimpleNamespace(
        root=root,
        out=out,
        music_scp=tmp_path / "music.scp",
        noise_scp=tmp_path / "noise.scp",
    )
    packs[0][3][0][3] = "http://creativecommons.org/publicdomain/zero/1.0/"
    write_lists(a, packs)
    music_keys = [ln.split()[0] for ln in a.music_scp.read_text().splitlines()]
    assert music_keys == ["fsd_00000", "music-x-0001", "slakh_00000"]
    assert a.noise_scp.read_text().split()[0] == "noise-y-0001"
    rows = list(csv.DictReader((out / "licences.tsv").open(), delimiter="\t"))
    by_clip = {r["clip"]: r for r in rows}
    assert by_clip["music-x-0001"]["licence"] == "CC BY 3.0"
    assert by_clip["music-x-0001"]["url"] == "http://a"
    assert "music-x-0002" not in by_clip and "c2" not in by_clip
    assert by_clip["t"]["licence"] == "CC BY 4.0" and "zenodo" in by_clip["t"]["url"]
    # --relist reads the packs back from licences.tsv
    again = read_packs(out / "licences.tsv")
    assert [(b, s, p.name) for b, s, p, _ in again] == [
        ("music", "fsd50k_music", "fsd_00000.wav"),
        ("music", "slakh2100", "slakh_00000.wav"),
    ]


def test_a_pack_left_out_once_can_be_kept_again(tmp_path):
    """packs.tsv keeps every pack: a relist with another filter can bring one back."""
    from types import SimpleNamespace

    import aug_banks
    from aug_banks import read_packs, write_lists

    root = tmp_path / "corpora"
    (root / "musan" / "music").mkdir(parents=True)
    (root / "musan" / "noise").mkdir(parents=True)
    (root / "chime3_background").mkdir()
    out = tmp_path / "banks"
    out.mkdir()
    pack = out / "music" / "openmic2018_00000.wav"
    badge = "http://i.creativecommons.org/p/zero/1.0/88x31.png"
    packs = [("music", "openmic2018", pack, [["k1", 0.0, 10.0, badge, "m1"]])]
    a = SimpleNamespace(
        root=root, out=out, music_scp=tmp_path / "m.scp", noise_scp=tmp_path / "n.scp"
    )
    real = aug_banks.clip_ok
    try:
        aug_banks.clip_ok = lambda source, licence: False  # a filter that drops it
        write_lists(a, packs)
    finally:
        aug_banks.clip_ok = real
    assert a.music_scp.read_text() == ""
    again = read_packs(out / "packs.tsv")
    assert [p.name for _, _, p, _ in again] == ["openmic2018_00000.wav"]
    write_lists(a, again)  # the current filter: the CC0 badge passes
    assert a.music_scp.read_text().split()[0] == "openmic2018_00000"
