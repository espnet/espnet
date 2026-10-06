"""Tests for the LibriSpeech-PC cross-sentence manifest builder."""

import os
from pathlib import Path

import pytest

from egs3.libritts.f5tts.local.prepare_librispeech_pc import (
    build_gt_wav_dir,
    build_manifest,
)

LST_ROW = (
    "4992-41806-0009\t4.355\texclaimed Bill Harmon to his wife.\t"
    "4992-23283-0000\t6.645\tBut the more forgetfulness had then prevailed.\n"
)


@pytest.fixture()
def fake_tree(tmp_path: Path) -> tuple[Path, Path]:
    lst = tmp_path / "meta.lst"
    lst.write_text(LST_ROW, encoding="utf-8")
    root = tmp_path / "test-clean"
    for utt in ("4992-41806-0009", "4992-23283-0000"):
        spk, chap, _ = utt.split("-")
        d = root / spk / chap
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{utt}.flac").write_bytes(b"fake")
    return lst, root


def test_build_manifest_resolves_paths(fake_tree, tmp_path):
    lst, root = fake_tree
    out = tmp_path / "manifest.tsv"
    n = build_manifest(lst, root, out)
    assert n == 1
    gen_utt, gen_text, ref_utt, ref_wav, ref_text = (
        out.read_text(encoding="utf-8").strip().split("\t")
    )
    assert gen_utt == "4992-23283-0000"
    assert gen_text == "But the more forgetfulness had then prevailed."
    assert ref_utt == "4992-41806-0009"
    assert ref_wav == str(root / "4992" / "41806" / "4992-41806-0009.flac")
    assert ref_text == "exclaimed Bill Harmon to his wife."


def test_build_manifest_missing_audio_raises(fake_tree, tmp_path):
    lst, root = fake_tree
    (root / "4992" / "41806" / "4992-41806-0009.flac").unlink()
    with pytest.raises(FileNotFoundError):
        build_manifest(lst, root, tmp_path / "manifest.tsv")


def test_build_gt_wav_dir_symlinks_targets(fake_tree, tmp_path):
    lst, root = fake_tree
    gt_dir = tmp_path / "gt_wavs"
    n = build_gt_wav_dir(lst, root, gt_dir)
    assert n == 1
    link = gt_dir / "4992-23283-0000.wav"
    assert link.is_symlink()
    assert (
        link.resolve() == (root / "4992" / "23283" / "4992-23283-0000.flac").resolve()
    )
    # Test with relative root: it should still resolve correctly
    rel_root = Path(os.path.relpath(root, Path.cwd()))
    gt_dir_rel = tmp_path / "gt_wavs_rel"
    n_rel = build_gt_wav_dir(lst, rel_root, gt_dir_rel)
    assert n_rel == 1
    link_rel = gt_dir_rel / "4992-23283-0000.wav"
    assert link_rel.is_symlink()
    assert (
        link_rel.resolve()
        == (root / "4992" / "23283" / "4992-23283-0000.flac").resolve()
    )
