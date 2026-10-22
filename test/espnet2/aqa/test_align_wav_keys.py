"""Test the align_wav_keys utility."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
UTILS = ROOT / "egs2/TEMPLATE/asr1/pyscripts/utils"


def test_align_keys(tmp_path):
    """Alignment preserves values and emits sorted union keys, ignoring blanks."""
    first, second, output = (tmp_path / name for name in ("first", "second", "out"))
    first.write_text("b target.wav\n\na target.wav\n")
    second.write_text("c cat ref.wav |\n\nb ref with spaces.wav\n")
    subprocess.run(
        [
            sys.executable,
            str(UTILS / "align_wav_keys.py"),
            str(first),
            str(second),
            str(output),
        ],
        check=True,
    )
    assert output.read_text() == "a None\nb ref with spaces.wav\nc cat ref.wav |\n"
