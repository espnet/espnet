"""Turn a native ``Speech2Text`` result into the serialized hypothesis."""

from __future__ import annotations


def render_sot(result) -> str:
    """Return the best hypothesis with its timestamps and separators.

    The text comes from the decoder's own converter, which this system's
    ``Speech2Text`` builds to keep special tokens, so a separator the
    checkpoint appended to its vocabulary survives. Only the prompt is
    removed: the language and task symbols the decoder was primed with.

    Args:
        result: The n-best list ``Speech2Text.__call__`` returns, each entry
            ``(text, tokens, token_ids, text_nospecial, hypothesis)``.

    Returns:
        The best hypothesis, for example
        ``"<|0.00|> hi<|1.20|> ???? <|0.50|> yes<|0.90|>"``; ``""`` when the
        decoder returned no hypothesis.

    Raises:
        ValueError: If the text does not start with the prompt tokens, which
            means it was rendered by a converter that drops special tokens.

    Examples:
        >>> tokens = ["<|en|>", "<|transcribe|>", "<|0.00|>", " hi", "<|0.40|>"]
        >>> render_sot([("".join(tokens), tokens, [], " hi", None)])
        '<|0.00|> hi<|0.40|>'
    """
    if not result:
        return ""
    text, tokens, *_ = result[0]
    prompt = "".join(tokens[:2])
    # A converter that drops special tokens leaves words in these two slots.
    special = [t for t in tokens[:2] if t.startswith("<|") and t.endswith("|>")]
    if len(special) != 2 or not text.startswith(prompt):
        raise ValueError(
            f"hypothesis {text!r} does not start with its prompt {prompt!r}; "
            "decode with espnet3.systems.esp2_s2t.src.s2t_inference.Speech2Text "
            "and speaker_change_symbol or adaptive_timestamp set"
        )
    return text[len(prompt) :]
