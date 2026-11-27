import io
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import soundfile as sf

from egs3.urgent25.snoar.dataset.builder import Urgent25Builder, write_dev_set
from egs3.urgent25.snoar.dataset.licenses import (
    audit_noise_licenses,
    parse_cc_license,
    write_license_record,
)

CC = "http://creativecommons.org/licenses/{}/3.0/"


def _write_store(root, subset, ids, sr=16000, seconds=0.5):
    """An OMNIIO subset: one blob of FLAC clips and its metadata.parquet."""
    d = root / subset
    d.mkdir(parents=True)
    blob, rows = d / "blob_0.bin", []
    with open(blob, "wb") as f:
        for k, i in enumerate(ids):
            buf = io.BytesIO()
            x = 0.1 * np.sin(np.arange(int(sr * seconds)) * (0.01 + k / 100))
            sf.write(buf, x.astype(np.float32), sr, format="FLAC")
            s0 = f.tell()
            f.write(buf.getvalue())
            rows.append(
                dict(
                    id=i,
                    sample_rate=sr,
                    samples=int(sr * seconds),
                    start_byte=s0,
                    end_byte=f.tell(),
                    path=str(blob),
                )
            )
    pq.write_table(pa.Table.from_pylist(rows), d / "metadata.parquet")


@pytest.fixture
def sources(tmp_path):
    """A tiny URGENT store plus FSD50K / FSDKaggle2018 / FMA / MUSAN metadata."""
    urgent = tmp_path / "urgent2025"
    _write_store(urgent, "noise_fsd50k", ["fsd50k_1", "fsd50k_2", "fsd50k_3"])
    _write_store(urgent, "noise_fma_novocal", ["fma_10", "fma_11"])
    _write_store(
        urgent,
        "noise_dns5",
        ["Freesound_validated_500_0", "Freesound_validated_600_0", "audioset_x"],
    )
    fsd = tmp_path / "FSD50K.metadata"
    fsd.mkdir()
    clips = {
        "1": {"license": CC.format("by"), "uploader": "ann", "title": "rain"},
        "2": {"license": CC.format("by-nc"), "uploader": "bob", "title": "car"},
        "3": {"license": "http://creativecommons.org/publicdomain/zero/1.0/"},
    }
    (fsd / "dev_clips_info_FSD50K.json").write_text(json.dumps(clips))
    (fsd / "eval_clips_info_FSD50K.json").write_text("{}")
    kag = tmp_path / "FSDKaggle2018.meta"
    kag.mkdir()
    (kag / "train_post_competition.csv").write_text(
        "fname,label,manually_verified,freesound_id,license\n"
        "a.wav,Bark,1,500,Attribution\n"
    )
    (kag / "test_post_competition_scoring_clips.csv").write_text(
        "fname,label,usage,freesound_id,license\n"
    )
    fma = tmp_path / "fma_metadata"
    fma.mkdir()
    (fma / "raw_tracks.csv").write_text(
        "track_id,artist_name,track_title,track_url,license_url\n"
        f"10,cat,song,http://fma/10,{CC.format('by-sa')}\n"
        "11,dog,tune,http://fma/11,http://creativecommons.org/publicdomain/mark/1.0/\n"
    )
    musan = tmp_path / "musan" / "noise"
    for sub, name in (
        ("free-sound", "noise-free-sound-0001"),
        ("sound-bible", "noise-sound-bible-0001"),
        ("sound-bible", "noise-sound-bible-0002"),
    ):
        (musan / sub).mkdir(parents=True, exist_ok=True)
        sf.write(musan / sub / f"{name}.wav", np.zeros(800, np.float32), 16000)
    (musan / "sound-bible" / "LICENSE").write_text(
        "noise-sound-bible-0001\nTitle: Bell\nLicense: Attribution 3.0\n"
        "Recorded by eve\nhttp://soundbible.com/1.html\n=====\n"
        "noise-sound-bible-0002\nTitle: Bang\nLicense: Noncommercial 3.0\n=====\n"
    )
    return dict(
        urgent_root=urgent,
        fsd50k_meta=fsd,
        fma_meta=fma,
        fsdkaggle_meta=kag,
        musan_noise=musan,
    )


def test_parse_cc_license():
    assert parse_cc_license(CC.format("by-nc-sa")) == "by-nc-sa"
    assert parse_cc_license(None) == "unknown"


def test_audit_keeps_only_commercial_use_licenses(sources, tmp_path):
    rows = audit_noise_licenses(**sources)
    counts = write_license_record(rows, tmp_path / "out")
    allow = (tmp_path / "out" / "commercial_allow.txt").read_text().split()
    # CC BY and CC0 FSD50K clips, the public-domain FMA track, the DNS5 clip
    # whose Freesound id FSDKaggle2018 lists as CC BY; not NC, SA, unknown or
    # AudioSet
    assert allow == [
        "noise_fsd50k/fsd50k_1",
        "noise_fsd50k/fsd50k_3",
        "noise_fma_novocal/fma_11",
        "noise_dns5/Freesound_validated_500_0",
    ]
    musan = (tmp_path / "out" / "musan_noise_commercial.txt").read_text().split()
    assert [m.rsplit("/", 1)[1] for m in musan] == [
        "noise-free-sound-0001.wav",
        "noise-sound-bible-0001.wav",
    ]
    assert counts == dict(files=11, allowed=4, musan=2)
    credit = (tmp_path / "out" / "noise_attribution.tsv").read_text().splitlines()
    assert "noise_fsd50k\tfsd50k_1\tby\tFSD50K clips_info\tann\train\t" in credit[1]
    assert "noise/sound-bible/noise-sound-bible-0001.wav\tby" in credit[-1]
    assert str(tmp_path) not in "\n".join(credit)


def test_audit_without_allowed_files_raises(sources, tmp_path):
    rows = [r for r in audit_noise_licenses(**sources) if r[2] == "by-nc"]
    with pytest.raises(ValueError, match="no noise file has a commercial-use"):
        write_license_record(rows, tmp_path / "out")


def test_builder_writes_lists_and_dev_set(sources, tmp_path):
    dev = tmp_path / "urgent2026"
    _write_store(dev, "dev_noisy", ["d0", "d1"], sr=48000)
    _write_store(dev, "dev_clean", ["d0", "d1"], sr=48000)
    out = tmp_path / "data"
    kwargs = dict(sources, dev_root=dev, out_dir=out)
    builder = Urgent25Builder()
    assert builder.is_source_prepared(**kwargs)
    assert not builder.is_built(**kwargs)
    builder.build(**kwargs)
    assert builder.is_built(**kwargs)
    rows = [json.loads(line) for line in open(out / "manifest/dev.jsonl")]
    assert [r["id"] for r in rows] == ["d0", "d1"]
    info = sf.info(rows[0]["speech_ref"])
    assert info.samplerate == 16000 and info.frames == 8000


def test_builder_errors(sources, tmp_path):
    builder = Urgent25Builder()
    assert not builder.is_source_prepared(urgent_root=None, dev_root=None)
    with pytest.raises(FileNotFoundError, match="URGENT stores not found"):
        builder.prepare_source(urgent_root="/x", dev_root=None)
    with pytest.raises(ValueError, match="fsd50k_meta and fma_meta must be set"):
        builder.build(urgent_root=sources["urgent_root"], out_dir=tmp_path / "o")
    dev = tmp_path / "urgent2026"
    _write_store(dev, "dev_noisy", ["d0", "d1"])
    _write_store(dev, "dev_clean", ["d0"])
    with pytest.raises(ValueError, match="lack a reference"):
        write_dev_set(dev, tmp_path / "o")
