#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""Reading hypotheses and references.

VERSA's ``--io`` chooses how waveforms are found (``kaldi``, ``soundfile``,
``dir``). SPLET's chooses how text is found, with the same flag and the same
shape of result: a dict from utterance key to content.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from splet.structures import Session, Turn

IO_CHOICES = ("kaldi", "jsonl", "dir")


def text_loader_setup(path: str, io: str = "kaldi") -> Dict[str, str]:
    """Load plain text keyed by utterance id.

    Args:
        path: File or directory to read, depending on ``io``.
        io: ``kaldi`` for a Kaldi ``text`` file (``uttid the rest of the
            line``); ``jsonl`` for one JSON object per line with a ``key``
            and a ``text`` field; ``dir`` for a directory of files, each
            named after its utterance id.

    Returns:
        Utterance id to text.

    Raises:
        ValueError: If ``io`` is unknown, a line cannot be parsed, or an
            utterance id occurs twice. A duplicate is never resolved by
            keeping one of the two: whichever won, the other's words would
            vanish from the count without anyone asking.
    """
    if io == "kaldi":
        entries: Dict[str, str] = {}
        with open(path, encoding="utf-8") as handle:
            for number, line in enumerate(handle, start=1):
                line = line.rstrip("\n")
                if not line.strip():
                    continue
                parts = line.split(maxsplit=1)
                key = parts[0]
                if key in entries:
                    raise ValueError(f"{path}:{number}: duplicate utterance id '{key}'")
                # A line with only an id is an utterance the system produced
                # nothing for. Dropping it would quietly remove its reference
                # words from the denominator and improve the result.
                entries[key] = parts[1] if len(parts) > 1 else ""
        return entries

    if io == "jsonl":
        return {
            key: session.text() for key, session in session_loader_setup(path).items()
        }

    if io == "dir":
        entries = {}
        sources: Dict[str, Path] = {}
        for child in sorted(Path(path).iterdir()):
            if not child.is_file():
                continue
            if child.stem in sources:
                raise ValueError(
                    f"{path}: '{sources[child.stem].name}' and '{child.name}' "
                    f"both name utterance '{child.stem}'"
                )
            sources[child.stem] = child
            entries[child.stem] = child.read_text(encoding="utf-8").strip()
        return entries

    raise ValueError(f"unknown io '{io}': expected one of {IO_CHOICES}")


def session_loader_setup(path: str) -> Dict[str, Session]:
    """Load structured, optionally speaker-attributed, sessions from JSONL.

    One JSON object per line::

        {"key": "meeting1", "turns": [
            {"speaker": "A", "start": 0.0, "end": 1.2, "text": "hello"},
            {"speaker": "B", "start": 1.0, "end": 2.4, "text": "hi there"}]}

    A line with a plain ``"text"`` and no ``"turns"`` is read as a
    single-turn session, so the same file format covers both tiers.

    Args:
        path: JSONL file to read.

    Returns:
        Session key to session.

    Raises:
        ValueError: If a line has no key, is neither shape, or repeats a key
            an earlier line used.
    """
    sessions: Dict[str, Session] = {}
    with open(path, encoding="utf-8") as handle:
        for number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            key = record.get("key", record.get("id", record.get("utt_id")))
            if key is None:
                raise ValueError(f"{path}:{number}: record has no 'key'")
            if key in sessions:
                raise ValueError(f"{path}:{number}: duplicate key '{key}'")
            if "turns" in record:
                turns = [
                    Turn(
                        text=turn.get("text", ""),
                        speaker=turn.get("speaker"),
                        start=turn.get("start"),
                        end=turn.get("end"),
                        extra={
                            field: value
                            for field, value in turn.items()
                            if field not in ("text", "speaker", "start", "end")
                        },
                    )
                    for turn in record["turns"]
                ]
                sessions[key] = Session(key=key, turns=turns)
            elif "text" in record:
                sessions[key] = Session.from_text(key, record["text"])
            else:
                raise ValueError(
                    f"{path}:{number}: record has neither 'turns' nor 'text'"
                )
    return sessions
