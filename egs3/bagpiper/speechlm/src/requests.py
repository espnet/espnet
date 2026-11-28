"""Read Bagpiper inference requests from a prepared SpeechLM dialogue split."""

import json
from pathlib import Path

from espnet3.components.data.base_dataset import BaseDataset
from espnet3.contract import Field
from espnet3.utils.scp_utils import check_utt_id


class Dataset(BaseDataset):
    """Select the user request before the first assistant turn in each dialogue.

    Args:
        manifest: Prepared SpeechLM JSON with a dialogue reader and sample IDs.
        mode: ``understand`` for audio and a question, or ``generate`` for text.

    Raises:
        ValueError: A selected ID is missing, repeated, or has an unsupported
            sequence of system and user turns.
    """

    def __init__(self, manifest: str, mode: str):
        if mode not in ("understand", "generate"):
            raise ValueError("mode must be 'understand' or 'generate'")
        manifest = Path(manifest).resolve()
        with manifest.open(encoding="utf-8") as stream:
            split = json.load(stream)
        entries = split.get("data_entry") if isinstance(split, dict) else None
        if not (
            isinstance(entries, list)
            and len(entries) == 1
            and isinstance(entries[0], dict)
            and entries[0].get("name") == entries[0].get("reader") == "dialogue"
            and isinstance(entries[0].get("path"), str)
            and entries[0]["path"]
        ):
            raise ValueError(f"{manifest} must declare one 'dialogue' JSONL reader")
        source = manifest.parent / entries[0]["path"]
        self.fields = (
            Field("utt_id", "text"),
            Field("prompt", "text"),
            Field("system", "text", optional=True),
        )
        if mode == "understand":
            self.fields += (Field("speech", "audio"),)

        requested = split.get("samples")
        if not isinstance(requested, list) or not all(
            isinstance(key, str) for key in requested
        ):
            raise ValueError(f"{manifest} must contain a 'samples' list of string IDs")
        selected = {check_utt_id(key) for key in requested}
        if len(requested) != len(selected):
            raise ValueError(f"Duplicate sample IDs in {manifest}")
        indexed = {}
        with source.open(encoding="utf-8") as stream:
            for line_number, line in enumerate(stream, 1):
                try:
                    row = json.loads(line)
                    key = row["example_id"]
                except (ValueError, TypeError, KeyError) as exc:
                    raise ValueError(
                        f"Invalid dialogue at {source}:{line_number}"
                    ) from exc
                if not isinstance(key, str):
                    raise ValueError(f"Invalid example_id at {source}:{line_number}")
                if key in selected:
                    if key in indexed:
                        raise ValueError(f"Duplicate dialogue {key!r} in {source}")
                    indexed[key] = self._get_request(
                        key, row.get("messages"), mode, source.parent
                    )
        missing = selected.difference(indexed)
        if missing:
            raise ValueError(f"Missing {len(missing)} dialogue(s) in {source}")
        self.requests = [indexed[key] for key in requested]

    @staticmethod
    def _get_request(key, messages, mode, directory):
        if not isinstance(messages, list):
            raise ValueError(f"Expected a messages list for {key!r}")
        prefix = []
        for turn in messages:
            if isinstance(turn, list) and turn[:1] == ["assistant"]:
                break
            if (
                not isinstance(turn, list)
                or len(turn) != 3
                or not all(isinstance(value, str) for value in turn)
            ):
                raise ValueError(f"Invalid request turn for {key!r}: {turn!r}")
            prefix.append(turn)

        request = {"utt_id": key}
        if prefix and prefix[0][:2] == ["system", "text"]:
            request["system"] = prefix.pop(0)[2]
        expected = [["user", "text"]]
        if mode == "understand":
            expected.insert(0, ["user", "audio"])
        if [turn[:2] for turn in prefix] != expected:
            raise ValueError(
                f"Unexpected {mode} request turns for {key!r}: expected {expected}"
            )
        if mode == "understand":
            if not prefix[0][2]:
                raise ValueError(f"Missing input audio path for {key!r}")
            request["speech"] = str(directory / prefix[0][2])
        request["prompt"] = prefix[-1][2]
        return request

    def __len__(self):
        return len(self.requests)

    def __getitem__(self, index):
        """Return one request in the order of the split's sample IDs."""
        return self.requests[index]
