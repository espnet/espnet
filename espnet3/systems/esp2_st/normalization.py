"""Text normalization for ST: case conventions and Moses punctuation rules.

ST recipes label each side of the corpus with a case convention and build a
separate text stream for it:

* ``tc``     truecased, punctuation kept (``egs2/must_c/st1`` target side)
* ``lc``     lowercased
* ``lc.rm``  lowercased with punctuation stripped (both sides of
  ``egs2/covost2/st1``, source side of ``egs2/must_c/st1``)

``local/data_prep.sh`` builds these with Moses' ``lowercase.perl`` and espnet's
``utils/remove_punctuation.pl``. Both are ported here so the recipes reproduce
the same character inventory instead of inventing their own normalization.

The Moses ``normalize-punctuation.perl`` / ``tokenizer.perl`` passes are
deliberately NOT reproduced: they are tokenization, which SentencePiece
replaces.
"""

import re
import string
import unicodedata

CASES = ("tc", "lc", "lc.rm")

# Perl's [[:punct:]] under Unicode rules matches \p{Punct} plus the ASCII
# symbols ($ + < = > ^ ` | ~), which Unicode classifies as Symbol.
_ASCII_PUNCT = frozenset(string.punctuation)


def is_punctuation(char: str) -> bool:
    """Return whether ``char`` is what Perl's ``[[:punct:]]`` would match."""
    return unicodedata.category(char).startswith("P") or char in _ASCII_PUNCT


def remove_punctuation(text: str) -> str:
    """Port of espnet's ``utils/remove_punctuation.pl``.

    Apostrophes survive, and the ``<space>`` scoring marker is protected, both
    via the same placeholder trick the Perl script uses.
    """
    text = text.replace("<space>", "spacemark").replace("'", "apostrophe")
    text = "".join(char for char in text if not is_punctuation(char))
    text = text.replace("apostrophe", "'").replace("spacemark", "<space>")
    return " ".join(text.split())


def apply_case(text: str, case: str) -> str:
    """Apply one of ``st.sh``'s case conventions to ``text``.

    Args:
        text: The raw transcript or translation.
        case: One of ``tc``, ``lc`` or ``lc.rm``.

    Returns:
        The text under that convention.

    Raises:
        ValueError: If ``case`` is not one of :data:`CASES`.
    """
    if case == "tc":
        return text
    if case == "lc":
        return text.lower()
    if case == "lc.rm":
        return remove_punctuation(text.lower())
    raise ValueError(f"Unknown case {case!r}; expected one of {CASES}")


# ---------------------------------------------------------------------------
# Moses normalize-punctuation.perl
# ---------------------------------------------------------------------------
# Called by egs2/must_c/st1/local/data_prep.sh and
# egs2/covost2/st1/local/data_prep_covost2.sh. This is character normalization,
# not tokenization, so unlike tokenizer.perl it has to be reproduced: skipping
# it leaves a different character inventory in the SentencePiece vocabulary.

_QUOTE_LETTER_APOS = (
    (re.compile(r"([a-z])‘([a-z])", re.IGNORECASE), r"\1'\2"),
    (re.compile(r"([a-z])’([a-z])", re.IGNORECASE), r"\1'\2"),
)


def normalize_punctuation(text: str, language: str = "en") -> str:
    """Port of Moses' ``normalize-punctuation.perl``.

    Args:
        text: One line of raw corpus text.
        language: Affects only the two language-conditional rules at the end
            (quote/comma ordering, and the digit-separator inserted between
            two space-separated digits).

    Returns:
        The normalized line.
    """
    text = text.replace("\r", "")

    # remove extra spaces
    text = text.replace("(", " (").replace(")", ") ")
    text = re.sub(r" +", " ", text)
    text = re.sub(r"\) ([.!:?;,])", r")\1", text)
    text = text.replace("( ", "(").replace(" )", ")")
    text = re.sub(r"(\d) %", r"\1%", text)
    text = text.replace(" :", ":").replace(" ;", ";")

    # normalize unicode punctuation
    text = text.replace("`", "'").replace("''", ' " ')
    text = text.replace("„", '"').replace("“", '"').replace("”", '"')
    text = text.replace("–", "-").replace("—", " - ")
    text = re.sub(r" +", " ", text)
    text = text.replace("´", "'")
    for pattern, replacement in _QUOTE_LETTER_APOS:
        text = pattern.sub(replacement, text)
    text = text.replace("‘", '"').replace("‚", '"').replace("’", '"')
    text = text.replace("''", '"').replace("´´", '"').replace("…", "...")

    # French quotes. NBSP, not ASCII space: Moses writes U+00A0 here, which
    # renders identically and silently became a plain space when transcribed.
    text = text.replace("\u00a0«\u00a0", ' "')
    text = text.replace("«\u00a0", '"').replace("«", '"')
    text = text.replace("\u00a0»\u00a0", '" ')
    text = text.replace("\u00a0»", '"').replace("»", '"')

    # Pseudo-spaces: every left-hand side here is U+00A0, and the right-hand
    # side an ordinary space or nothing. Written as escapes because the two are
    # indistinguishable on screen -- with ASCII on the left, four of these
    # rules replace a string with itself and the rest strip spaces Moses keeps.
    text = text.replace("\u00a0%", "%")
    text = text.replace("nº\u00a0", "nº ")
    text = text.replace("\u00a0:", ":")
    text = text.replace("\u00a0ºC", " ºC")
    text = text.replace("\u00a0cm", " cm")
    text = text.replace("\u00a0?", "?")
    text = text.replace("\u00a0!", "!")
    text = text.replace("\u00a0;", ";")
    text = text.replace(",\u00a0", ", ")
    text = re.sub(r" +", " ", text)

    if language == "en":
        text = re.sub(r'"([,.]+)', r'\1"', text)
    elif language not in {"cs", "cz"}:
        text = text.replace(',"', '",')
        text = re.sub(r'(\.+)"(\s*[^<])', r'"\1\2', text)

    if language in {"de", "es", "cz", "cs", "fr"}:
        text = re.sub(r"(\d) (\d)", r"\1,\2", text)
    else:
        text = re.sub(r"(\d) (\d)", r"\1.\2", text)
    return text
