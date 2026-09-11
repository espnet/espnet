"""Regression test: distinct dataset IDs that sanitize to the same utt_id must
fail preparation loudly (naming both source IDs), not silently drop one.

Self-contained: it builds a tiny DatasetDict fixture in a temp dir, so it does
not need the (undistributed) Nahuatl corpus and always runs.
"""
import os
import subprocess
import sys
import tempfile

from datasets import Dataset, DatasetDict, Features, Value

SCRIPT = os.path.join(os.path.dirname(__file__), "data_prep.py")

# Both ids share speaker code JBM566 and _sanitize()s to "JBM566_a_b", so both
# map to utt_id "JBM566_JBM566_a_b" while being distinct recordings.
_COLLIDING_IDS = ["JBM566_a-b", "JBM566_a_b"]


def _build_fixture(path):
    feats = Features(
        {
            "id": Value("string"),
            "text": Value("string"),
            "audio": {"bytes": Value("binary"), "path": Value("string")},
        }
    )
    data = {
        "id": _COLLIDING_IDS,
        "text": ["hola", "adios"],
        "audio": [
            {"bytes": b"RIFF0000", "path": ""},
            {"bytes": b"RIFF1111", "path": ""},
        ],
    }
    dd = DatasetDict({"hidalgo-train": Dataset.from_dict(data, features=feats)})
    dd.save_to_disk(path)


def test_sanitization_collision_fails_loudly():
    with tempfile.TemporaryDirectory() as tmpdir:
        hf_dir = os.path.join(tmpdir, "hf_data")
        _build_fixture(hf_dir)
        result = subprocess.run(
            [
                sys.executable, SCRIPT,
                "--hf_data_dir", hf_dir,
                "--split", "hidalgo-train",
                "--output_dir", os.path.join(tmpdir, "kaldi"),
                "--wav_dir", os.path.join(tmpdir, "wav"),
                "--region_token", "<nah_hid>",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0, "collision should fail, not be dropped"
        # both distinct source IDs are named so the user can disambiguate
        for raw_id in _COLLIDING_IDS:
            assert raw_id in result.stderr, f"{raw_id} not reported: {result.stderr}"
