"""Splitting a decoded SLU sequence into its intent and its transcript."""

from __future__ import annotations

from typing import Tuple


def split_intent(text: str) -> Tuple[str, str]:
    """Split ``"<intent> <transcript>"`` into its two parts.

    ``SLUTask`` predicts the intent as a reserved symbol prefixed to the
    transcript, so the first token of a hypothesis is the label and the rest
    is the words.

    Args:
        text: A hypothesis whose first token is the intent label.

    Returns:
        ``(intent, transcript)``. The transcript is empty when the text holds
        only a label, and both parts are empty for empty text -- a short
        hypothesis must not make the ``measure`` stage fail.

    Examples:
        >>> split_intent("news_query read the news")
        ('news_query', 'read the news')
        >>> split_intent("news_query")
        ('news_query', '')
        >>> split_intent("")
        ('', '')
    """
    parts = text.strip().split(maxsplit=1)
    if not parts:
        return "", ""
    if len(parts) == 1:
        return parts[0], ""
    return parts[0], parts[1]
