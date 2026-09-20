#!/usr/bin/env python3

# Copyright 2026 ESPnet Developers
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

"""The structured-text types SPLET scores.

VERSA has no equivalent module because audio is always the same shape: a
waveform and a sample rate. Text is not. A single ESPnet3 system may emit a
bare string, a string with a timestamp, a string attributed to a speaker, or
a whole session of such turns, and the metric that applies depends on which.

Two shapes cover the tasks listed in espnet/espnet#6760:

``Turn``
    one piece of text, optionally attributed to a speaker and optionally
    placed on a timeline.
``Session``
    the turns belonging to one recording, in the order they were produced.

A plain utterance is the degenerate case: a session of one turn with no
speaker and no timestamps. That is deliberate. It means the long-form and
multi-speaker paths are the general ones and conventional WER is the special
case, rather than the other way around, which is what the issue asks for.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Turn:
    """One attributed, optionally timed piece of text."""

    text: str
    speaker: Optional[str] = None
    start: Optional[float] = None
    end: Optional[float] = None
    # Task-specific payload (intent, slots, entities, parses, ...). SPLET
    # carries it untouched so that structured-prediction metrics can read it
    # without every intermediate layer having to know the schema.
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def duration(self) -> Optional[float]:
        """Return the turn duration, or None if it is not on a timeline."""
        if self.start is None or self.end is None:
            return None
        return self.end - self.start


@dataclass
class Session:
    """The turns of one recording."""

    key: str
    turns: List[Turn] = field(default_factory=list)

    @classmethod
    def from_text(cls, key: str, text: str) -> "Session":
        """Build a single-turn session from a bare string."""
        return cls(key=key, turns=[Turn(text=text)])

    @property
    def speakers(self) -> List[str]:
        """Return the distinct speaker labels, in order of first appearance."""
        seen: List[str] = []
        for turn in self.turns:
            if turn.speaker is not None and turn.speaker not in seen:
                seen.append(turn.speaker)
        return seen

    def text(self, speaker: Optional[str] = None, sep: str = " ") -> str:
        """Concatenate the session's text, optionally for one speaker only.

        Args:
            speaker: If given, keep only turns attributed to that speaker.
            sep: Separator placed between turns.

        Returns:
            The concatenated transcript. This is what cpWER scores, and the
            reason concatenation lives here rather than in a metric: every
            long-form metric needs it and they must all do it identically.
        """
        turns = (
            self.turns
            if speaker is None
            else [t for t in self.turns if t.speaker == speaker]
        )
        return sep.join(t.text for t in turns if t.text)

    def is_timed(self) -> bool:
        """Return True if every turn carries both a start and an end time."""
        return all(t.start is not None and t.end is not None for t in self.turns)
