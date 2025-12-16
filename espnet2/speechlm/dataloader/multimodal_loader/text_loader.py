#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Text data loading utilities supporting plain and JSONL formats."""

import json
import logging
from pathlib import Path
from typing import Iterator, Tuple

try:
    from arkive.text.write_utils import _decompress_text_data
except ImportError:
    raise ImportError(
        "arkive is not installed. Install at https://github.com/wanchichen/arkive"
    )

try:
    import duckdb
except ImportError:
    raise ImportError(
        "duckdb is not installed. Please install it with: pip install duckdb"
    )


class ArkiveTextReader:
    """Dict-like lazy text reader using arkive parquets.

    Reads compressed text data from arkive parquet files. Text is stored in
    binary format with compression and accessed via byte offsets.

    Args:
        parquet_path: Path to the parquet file containing text metadata.
        valid_ids: List of valid IDs to keep (optional, keeps all if None).
        worker_id: Partition IDs by worker (optional, keeps all if None).
        world_size: Used for worker partitioning.
    """

    def __init__(
        self,
        parquet_path: str,
        valid_ids: list = None,
        worker_id: int = None,
        world_size: int = None,
    ):

        query = f"SELECT * FROM read_parquet('{parquet_path}')"
        result = duckdb.query(query)

        # filter query result before loading to df
        # avoids loading the whole query result into memory
        if valid_ids is not None:
            result = duckdb.query(
                f"""
                SELECT * FROM result
                WHERE utt_id IN ({','.join(f"'{id}'" for id in valid_ids)})
                 """
            )

        if worker_id is not None:
            assert (
                world_size is not None
            ), f"filtering by worker_id requires world_size, got {world_size}"
            result = duckdb.query(
                f"""
                SELECT * FROM result
                QUALIFY (row_number() OVER (ORDER BY utt_id) - 1)
                % {world_size} = {worker_id}
            """
            )

        self.data = result.pl()
        self.index = {
            utt_id: idx for idx, utt_id in enumerate(self.data["utt_id"].to_list())
        }

    def __getitem__(self, key: str) -> str:
        """Get text by ID."""
        idx = self.index[key]
        row = self.data.row(idx, named=True)

        bin_path = row["path"]
        start_offset = row["start_byte_offset"]
        file_size = row["file_size_bytes"]

        with open(bin_path, "rb") as f:
            f.seek(start_offset)
            data_bytes = f.read(file_size)

        text = _decompress_text_data(data_bytes)

        return text

    def __contains__(self, key: str) -> bool:
        """Check if ID exists in manifest."""
        return key in self.index

    def __len__(self) -> int:
        """Return number of items in manifest."""
        return len(self.data)

    def keys(self) -> Iterator[str]:
        """Return iterator over IDs."""
        return iter(self.index.keys())

    def values(self) -> Iterator[str]:
        """Return iterator over text values."""
        for key in self.index:
            yield self[key]

    def items(self) -> Iterator[Tuple[str, str]]:
        """Return iterator over (id, text) pairs."""
        for key in self.index:
            yield key, self[key]


class TextReader:
    """Dict-like text reader supporting plain and JSONL formats.

    Plain format: <id> <text content>
    JSONL format: {"id": "<id>", "text": "<text content>"}

    Format is determined by file suffix (.jsonl for JSONL, otherwise plain).

    Args:
        text_file: Path to text file (plain or JSONL format)
        valid_ids: List of valid IDs to keep (optional, keeps all if None)
    """

    def __init__(self, text_file: str, valid_ids: list = None):
        self.data = {}
        text_path = Path(text_file)

        # Determine format by file suffix
        is_jsonl = text_path.suffix == ".jsonl"

        # Convert valid_ids to set for faster lookup
        valid_ids_set = set(valid_ids) if valid_ids is not None else None

        # Load and filter lines
        with open(text_file, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue

                if is_jsonl:
                    item = json.loads(line)
                    if "id" not in item or "text" not in item:
                        logging.warning(
                            f"Skipping line {line_idx}: missing 'id' or " f"'text' key"
                        )
                        continue
                    example_id = item["id"]
                    content = item["text"]
                else:
                    parts = line.split(maxsplit=1)
                    if len(parts) == 2:
                        example_id, content = parts
                    else:
                        continue

                # Only keep if in valid_ids (or if valid_ids is None)
                if valid_ids_set is None or example_id in valid_ids_set:
                    self.data[example_id] = content

    def __getitem__(self, key: str) -> str:
        """Get text by ID."""
        return self.data[key]

    def __contains__(self, key: str) -> bool:
        """Check if ID exists."""
        return key in self.data

    def __len__(self) -> int:
        """Return number of items."""
        return len(self.data)

    def keys(self):
        """Return iterator over IDs."""
        return self.data.keys()

    def values(self):
        """Return iterator over texts."""
        return self.data.values()

    def items(self):
        """Return iterator over (id, text) pairs."""
        return self.data.items()
