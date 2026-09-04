import subprocess
from pathlib import Path

import pytest

from egs3.voxceleb.spk.dataset import builder


def make_tree(root: Path, relative_paths: list[str]) -> None:
    """Create empty audio files at the given paths under `root`."""
    for relative in relative_paths:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()


def test_audio_suffixes_lists_wav_first_then_the_convertible_ones():
    suffixes = builder.audio_suffixes()

    assert suffixes[0] == builder.AUDIO_SUFFIX
    assert ".m4a" in suffixes


def test_scan_split_indexes_wav_and_aac_alike(tmp_path):
    make_tree(
        tmp_path,
        [
            "id10002/vidB/utt2.wav",
            "id10001/vidA/utt1.m4a",
        ],
    )

    entries = builder.scan_split(tmp_path)

    # Sorted by utterance ID, whatever the container.
    assert [utt_id for utt_id, _, _ in entries] == [
        "id10001/vidA/utt1",
        "id10002/vidB/utt2",
    ]
    assert [speaker for _, speaker, _ in entries] == ["id10001", "id10002"]
    assert [path.suffix for _, _, path in entries] == [".m4a", ".wav"]


def test_scan_split_rejects_a_tree_with_no_readable_audio(tmp_path):
    make_tree(tmp_path, ["id10001/vidA/utt1.flac"])

    with pytest.raises(RuntimeError, match="No audio found"):
        builder.scan_split(tmp_path)


def test_plan_conversions_passes_wav_through_and_redirects_the_rest(tmp_path):
    convert_root = tmp_path / "converted"
    entries = [
        ("id1/v/u1", "id1", Path("/corpus/id1/v/u1.wav")),
        ("id2/v/u2", "id2", Path("/corpus/id2/v/u2.m4a")),
    ]

    resolved, jobs = builder.plan_conversions(convert_root, "vox2_dev", entries)

    # The WAV entry is untouched; the AAC entry now points at its future WAV.
    assert resolved[0] == entries[0]
    assert resolved[1][2] == convert_root / "vox2_dev/id2/v/u2.wav"
    assert jobs == [
        ("/corpus/id2/v/u2.m4a", str(convert_root / "vox2_dev/id2/v/u2.wav"))
    ]


def test_plan_conversions_skips_files_that_were_already_converted(tmp_path):
    convert_root = tmp_path / "converted"
    target = convert_root / "vox2_dev/id2/v/u2.wav"
    target.parent.mkdir(parents=True)
    target.touch()
    entries = [("id2/v/u2", "id2", Path("/corpus/id2/v/u2.m4a"))]

    resolved, jobs = builder.plan_conversions(convert_root, "vox2_dev", entries)

    # An interrupted conversion resumes instead of redoing finished work.
    assert resolved[0][2] == target
    assert jobs == []


def test_run_conversions_does_nothing_when_there_is_nothing_to_convert():
    # Must not reach Dask, so that an all-WAV corpus needs no parallel backend.
    builder.run_conversions([])


def test_run_conversions_explains_a_missing_ffmpeg(monkeypatch):
    monkeypatch.setattr(builder.shutil, "which", lambda _name: None)

    with pytest.raises(RuntimeError, match="ffmpeg is not on"):
        builder.run_conversions([("in.m4a", "out.wav")])


def test_convert_audio_renames_the_finished_file_into_place(tmp_path, monkeypatch):
    target = tmp_path / "out" / "utt.wav"

    def fake_run(command, **_kwargs):
        # ffmpeg writes the temporary file; the builder does the rename.
        Path(command[-1]).write_bytes(b"RIFF")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(builder.subprocess, "run", fake_run)

    written = builder.convert_audio(("in.m4a", str(target)), 16000, 1)

    assert Path(written) == target
    assert target.read_bytes() == b"RIFF"
    assert not list(target.parent.glob("*.partial"))


def test_convert_audio_raises_and_cleans_up_when_ffmpeg_fails(tmp_path, monkeypatch):
    target = tmp_path / "out" / "utt.wav"

    def fake_run(command, **_kwargs):
        Path(command[-1]).write_bytes(b"truncated")
        return subprocess.CompletedProcess(command, 1, "", "moov atom not found")

    monkeypatch.setattr(builder.subprocess, "run", fake_run)

    with pytest.raises(RuntimeError, match="moov atom not found"):
        builder.convert_audio(("in.m4a", str(target)), 16000, 1)

    # No truncated WAV may survive, or the next run would skip the file.
    assert not target.exists()
    assert not list(target.parent.glob("*.partial"))


def test_convert_audio_asks_ffmpeg_for_the_configured_format(tmp_path, monkeypatch):
    seen = {}

    def fake_run(command, **_kwargs):
        seen["command"] = command
        Path(command[-1]).write_bytes(b"RIFF")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(builder.subprocess, "run", fake_run)
    builder.convert_audio(("in.m4a", str(tmp_path / "utt.wav")), 16000, 1)

    command = seen["command"]
    assert command[0] == "ffmpeg"
    assert command[command.index("-ar") + 1] == "16000"
    assert command[command.index("-ac") + 1] == "1"
    assert command[command.index("-f") + 1] == "wav"


def test_write_spk2utt_groups_utterances_and_counts_speakers(tmp_path):
    entries = [
        ("id1/v/u1", "id1", Path("/a.wav")),
        ("id1/v/u2", "id1", Path("/b.wav")),
        ("id2/v/u1", "id2", Path("/c.wav")),
    ]

    n_speakers = builder.write_spk2utt(tmp_path, entries)

    assert n_speakers == 2
    assert (tmp_path / "spk2utt").read_text() == (
        "id1 id1/v/u1 id1/v/u2\nid2 id2/v/u1\n"
    )


def test_write_spk2utt_is_the_only_file_a_label_space_needs(tmp_path):
    builder.write_spk2utt(tmp_path, [("id1/v/u1", "id1", Path("/a.wav"))])

    # A speaker union is a label space, not a split: no wav.scp, no utt2spk.
    assert [f.name for f in tmp_path.iterdir()] == ["spk2utt"]


def test_write_manifests_still_writes_every_manifest_of_a_real_split(tmp_path):
    builder.write_manifests(tmp_path, [("id1/v/u1", "id1", Path("/a.wav"))])

    assert sorted(f.name for f in tmp_path.iterdir()) == sorted(builder.MANIFESTS)


def test_speaker_unions_only_reference_real_source_splits():
    sources = set(builder._CFG["sources"])

    for union, parts in builder._CFG["speaker_unions"].items():
        assert union not in sources, f"{union} shadows a real split"
        for part in parts:
            assert str(part) in sources, f"{union} references unknown split {part}"


def test_a_speaker_union_is_not_loadable_as_a_dataset_split():
    from egs3.voxceleb.spk.dataset import dataset as vox_dataset

    # Training on both dev sets lists them under `dataset.train` and lets
    # CombinedDataset merge them, so the union must not look like a split.
    for union in builder._CFG["speaker_unions"]:
        assert union not in vox_dataset._KNOWN_SPLITS
