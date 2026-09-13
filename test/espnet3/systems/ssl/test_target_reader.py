"""Tests for the index-keyed BEATs target reader."""

import multiprocessing
import pickle

import pytest
import torch

from espnet3.systems.ssl.target_reader import BeatsTargetReader

# ===============================================================
# Test Case Summary
# ===============================================================
#
# | Test Name                                  | Description                     |
# |--------------------------------------------|---------------------------------|
# | test_reads_targets_by_index_in_any_order   | Lines keyed by index, unsorted. |
# | test_survives_pickling_after_reads         | Open handle is not pickled.     |
# | test_forked_workers_read_correct_lines     | Workers forked after a parent   |
# |                                            | read do not share file offsets. |
# | test_line_without_tokens_reads_empty       | `<idx>` alone -> empty target.  |
# | test_rejects_mismatched_targets            | Missing / duplicate / out of    |
# |                                            | range / malformed index raise.  |
# | test_missing_file_raises                   | Nonexistent path raises.        |


def _write(path, lines):
    path.write_text("".join(f"{line}\n" for line in lines), encoding="utf-8")
    return path


def test_reads_targets_by_index_in_any_order(tmp_path):
    path = _write(tmp_path / "target.scp", ["2 7 8", "0 1 2 3", "1 4"])

    reader = BeatsTargetReader(path, num_items=3)

    assert len(reader) == 3
    assert [reader[i] for i in range(3)] == ["1 2 3", "4", "7 8"]


def test_survives_pickling_after_reads(tmp_path):
    path = _write(tmp_path / "target.scp", ["0 5 6", "1 7"])
    reader = BeatsTargetReader(path, num_items=2)
    assert reader[1] == "7"

    restored = pickle.loads(pickle.dumps(reader))

    assert restored[0] == "5 6"


class _ReaderDataset(torch.utils.data.Dataset):
    def __init__(self, reader):
        self.reader = reader

    def __len__(self):
        return len(self.reader)

    def __getitem__(self, idx):
        return idx, self.reader[idx]


@pytest.mark.skipif(
    "fork" not in multiprocessing.get_all_start_methods(), reason="needs fork"
)
def test_forked_workers_read_correct_lines(tmp_path):
    num_items = 2000

    def expected_target(idx):
        return " ".join(str(idx % 97) for _ in range(idx % 13 + 50))

    path = _write(
        tmp_path / "target.scp",
        [f"{i} {expected_target(i)}" for i in range(num_items)],
    )
    reader = BeatsTargetReader(path, num_items=num_items)
    assert reader[0] == expected_target(0)  # parent opens a descriptor first
    loader = torch.utils.data.DataLoader(
        _ReaderDataset(reader),
        batch_size=None,
        shuffle=True,
        num_workers=4,
        multiprocessing_context="fork",
    )

    mismatches = [idx for idx, target in loader if target != expected_target(idx)]

    assert mismatches == []


def test_line_without_tokens_reads_empty(tmp_path):
    path = _write(tmp_path / "target.scp", ["0", "1 4"])

    reader = BeatsTargetReader(path, num_items=2)

    assert reader[0] == ""
    assert reader[1] == "4"


@pytest.mark.parametrize(
    "lines, match",
    [
        (["0 1"], "no target for 1"),
        (["0 1", "0 2"], "duplicate index 0"),
        (["0 1", "2 3"], "out of range"),
        (["utt0 1", "1 2"], "expected an integer"),
    ],
)
def test_rejects_mismatched_targets(tmp_path, lines, match):
    path = _write(tmp_path / "target.scp", lines)

    with pytest.raises(ValueError, match=match):
        BeatsTargetReader(path, num_items=2)


def test_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="infer"):
        BeatsTargetReader(tmp_path / "missing.scp", num_items=1)
