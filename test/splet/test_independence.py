"""SPLET must stay separable from the rest of the repository."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def test_splet_imports_nothing_from_espnet():
    """Run the CI check here too, so the failure is local and immediate.

    splet/ is meant to be extracted into its own repository
    (espnet/espnet#6760), which is a directory move only while nothing inside
    it reaches into ESPnet. The espnet3 metrics it will replace do exactly
    that today, so this is a live hazard rather than a hypothetical one.
    """
    checker = ROOT / "ci" / "check_splet_independence.py"
    result = subprocess.run(
        [sys.executable, str(checker)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stdout + result.stderr
