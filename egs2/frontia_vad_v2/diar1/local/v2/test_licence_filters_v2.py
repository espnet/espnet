"""Clip filters that keep material of unknown licence out of the dumps."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import adapters_farfield_read as afr  # noqa: E402
import adapters_nonspeech as ans  # noqa: E402


def test_starss23_music_clip_is_detected():
    with_music = "10,1,0,0,0\n11,8,0,0,0\n"
    without = "10,1,0,0,0\n11,0,1,0,0\n"
    assert afr.starss23_has_music(with_music)
    assert not afr.starss23_has_music(without)
    assert not afr.starss23_has_music("frame,class\n")


def test_dcase2025_keeps_only_known_provenance():
    assert ans.dcase2025_clip_kept("YH_0001.wav")
    assert ans.dcase2025_clip_kept("Background_0001.wav")
    assert not ans.dcase2025_clip_kept("bird_0001.wav")
    assert not ans.dcase2025_clip_kept("Bird_0001.wav")
