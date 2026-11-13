"""The batched alignment gives what the one-by-one alignment gives.

    LIBRISPEECH_DIR=<chapter dir> python3 -m pytest local/v2/test_align_batch.py

Needs OWSM-CTC v4 (Hugging Face cache or network) and a LibriSpeech chapter
directory (*.flac and the .trans.txt); skipped otherwise. Writes two dumps of
the same recordings (some packed, so of 1 to 3 buffers), aligns one with
--batch_seconds 0 and one batched, and compares the segments, scores, labels
and dropped lists.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import soundfile

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from vad_dump import VadDumpWriter  # noqa: E402

REVISION = "6075bf3d42e6037d1ef7abd7c5558ade886e6eaa"
CHAPTER = os.environ.get("LIBRISPEECH_DIR")


def _dump(path: Path, chapter: Path, n: int = 8):
    trans = next(chapter.glob("*.trans.txt"))
    w = VadDumpWriter(path, "ls", "read", "A", "CC BY 4.0", pack_seconds=25)
    for line in trans.read_text().splitlines()[:n]:
        uid, txt = line.split(maxsplit=1)
        x, rate = soundfile.read(chapter / f"{uid}.flac", dtype="float32")
        w.add(uid, x, rate, text=txt, lang_sym="<eng>")
    # a transcript that cannot match: aligned or dropped alike on both paths
    w.add("mismatch", x[:16000], rate, text="word " * 200, lang_sym="<eng>")
    w.close()


def _align(path: Path, batch_seconds: float):
    r = subprocess.run(
        [
            sys.executable,
            str(HERE / "align.py"),
            "--dump",
            str(path),
            "--revision",
            REVISION,
            "--device",
            "cuda" if os.environ.get("ALIGN_TEST_CUDA") else "cpu",
            "--batch_seconds",
            str(batch_seconds),
            "--workers",
            "2",
        ],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0, r.stderr[-2000:]


@pytest.mark.skipif(not CHAPTER, reason="LIBRISPEECH_DIR is not set")
def test_batched_equals_one_by_one(tmp_path):
    one, many = tmp_path / "one", tmp_path / "many"
    _dump(one, Path(CHAPTER))
    shutil.copytree(one, many)
    _align(one, 0)
    _align(many, 120)  # 4 buffers per forward: several recordings per batch
    a = (one / "align.tsv").read_text().splitlines()
    b = (many / "align.tsv").read_text().splitlines()
    assert len(a) == len(b) > 1
    for x, y in zip(a[1:], b[1:]):
        rx, ry = x.split("\t"), y.split("\t")
        assert rx[:4] == ry[:4]  # recording, word, start, end
        assert abs(float(rx[4]) - float(ry[4])) < 1e-3  # score
    for name in ("espnet_rttm", "align_dropped.tsv", "wav.scp", "manifest.tsv"):
        assert (one / name).read_text() == (many / name).read_text(), name
