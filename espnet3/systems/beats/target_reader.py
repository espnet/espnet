"""Index-keyed reader for BEATs token targets written by the ``infer`` stage."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class BeatsTargetReader:
    """Random-access reader for ``target.scp`` files keyed by dataset index.

    The BEATs ``infer`` stage writes one line per dataset item::

        <idx> <id> <id> <id> ...

    where ``<idx>`` is the integer index of the item in the dataset that was
    tokenized. Recipe datasets use this reader to attach a ``"target"`` string
    to the sample with the same index, which keeps samples free of ``utt_id``.

    Only byte offsets and lengths are kept in memory (12 bytes per item), so a
    2M-utterance AudioSet target file does not have to be loaded as Python
    strings in every DataLoader worker. Lines are read with ``os.pread`` from a
    file descriptor opened once per process, so forked DataLoader workers never
    share a file position.

    Args:
        target_path: Path to an index-keyed ``target.scp`` file.
        num_items: Expected number of dataset items. Every index in
            ``[0, num_items)`` must appear exactly once.

    Raises:
        FileNotFoundError: If ``target_path`` does not exist.
        ValueError: If an index is malformed, out of range, duplicated, or
            missing, which means the targets were produced for a different
            dataset (split, manifest order, or filtering).

    Examples:
        >>> reader = BeatsTargetReader("exp/beats_iter0/targets/train/target.scp",
        ...                            num_items=len(dataset))  # doctest: +SKIP
        >>> reader[0]  # doctest: +SKIP
        '886 468 468 ...'
    """

    def __init__(self, target_path: str | Path, num_items: int) -> None:
        """Index the byte offset of every target line."""
        self.target_path = Path(target_path)
        if not self.target_path.is_file():
            raise FileNotFoundError(
                f"Target file not found: {self.target_path}. Run the `infer` stage "
                "for this iteration first."
            )
        self.num_items = int(num_items)
        self._offsets, self._lengths = self._build_index()
        self._fd = None
        self._fd_pid = None

    def _build_index(self) -> tuple[np.ndarray, np.ndarray]:
        r"""Scan ``target_path`` once and locate the token ids of every index.

        Lines may appear in any order (``infer`` shards finish out of order),
        so the arrays are indexed by the ``<idx>`` key of each line, not by
        line number. For each index this records where its token ids start
        (after ``"<idx> "``) and how many bytes they span, including the
        newline that ``__getitem__`` strips.

        Returns:
            tuple[np.ndarray, np.ndarray]: ``(offsets, lengths)`` of shape
            ``(num_items,)``. For the file::

                1 7 8
                0 5

            they are ``offsets = [8, 2]`` and ``lengths = [2, 4]``: index 0
            is ``b"5\n"`` at byte 8 and index 1 is ``b"7 8\n"`` at byte 2.

        Raises:
            ValueError: If a line has a malformed, out-of-range, or duplicate
                index, or if some index in ``[0, num_items)`` has no line.
        """
        offsets = np.full(self.num_items, -1, dtype=np.int64)
        lengths = np.zeros(self.num_items, dtype=np.int32)
        offset = 0
        with self.target_path.open("rb") as handle:
            for line_number, line in enumerate(handle, start=1):
                key, _, _ = line.partition(b" ")
                try:
                    idx = int(key)
                except ValueError as err:
                    raise ValueError(
                        f"{self.target_path}:{line_number}: expected an integer "
                        f"dataset index, got {key[:32]!r}."
                    ) from err
                if not 0 <= idx < self.num_items:
                    raise ValueError(
                        f"{self.target_path}:{line_number}: index {idx} is out of "
                        f"range for a dataset of {self.num_items} items."
                    )
                if offsets[idx] != -1:
                    raise ValueError(
                        f"{self.target_path}:{line_number}: duplicate index {idx}."
                    )
                start = min(len(key) + 1, len(line))
                offsets[idx] = offset + start
                lengths[idx] = len(line) - start
                offset += len(line)
        missing = np.flatnonzero(offsets == -1)
        if missing.size:
            raise ValueError(
                f"{self.target_path} has no target for {missing.size} of "
                f"{self.num_items} items (first missing index: {missing[0]}). "
                "The targets were produced for a different dataset."
            )
        return offsets, lengths

    def __len__(self) -> int:
        """Return the number of indexed targets."""
        return self.num_items

    def __getitem__(self, idx: int) -> str:
        """Return the space-separated token ids for dataset item ``idx``.

        Examples:
            With the ``target.scp``::

                1 7 8
                0 5

            >>> reader = BeatsTargetReader("target.scp", num_items=2)  # doctest: +SKIP
            >>> reader[0], reader[1]  # doctest: +SKIP
            ('5', '7 8')
        """
        pid = os.getpid()
        if self._fd is None or self._fd_pid != pid:
            # A descriptor inherited through fork belongs to the parent; open
            # a private one so workers never race on shared state.
            self._fd = os.open(self.target_path, os.O_RDONLY)
            self._fd_pid = pid
        data = os.pread(self._fd, int(self._lengths[idx]), int(self._offsets[idx]))
        return data.decode("utf-8").strip()

    def __getstate__(self):
        """Drop the process-local file descriptor when pickling."""
        state = self.__dict__.copy()
        state["_fd"] = None
        state["_fd_pid"] = None
        return state
