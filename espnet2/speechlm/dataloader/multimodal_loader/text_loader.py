#!/usr/bin/env python3
# Copyright 2025 Jinchuan Tian (Carnegie Mellon University)
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Text data loading utilities supporting plain and JSONL formats."""

import json
import logging
from pathlib import Path


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
