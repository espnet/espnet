from pathlib import Path

import pytest

from espnet2.fileio.datadir_writer import DatadirWriter


def test_DatadirWriter(tmp_path: Path):
    writer = DatadirWriter(tmp_path)
    # enter(), __exit__(), close()
    with writer as f:
        # __getitem__()
        sub = f["aa"]
        # __setitem__()
        sub["bb"] = "aa"

        with pytest.raises(TypeError):
            sub["bb"] = 1
        with pytest.raises(RuntimeError):
            # Already has children
            f["aa"] = "dd"
        with pytest.raises(RuntimeError):
            # Is a text
            sub["cc"]

        # Create a directory, but set mismatched ids
        f["aa2"]["ccccc"] = "aaa"
        # Duplicated warning
        f["aa2"]["ccccc"] = "def"
