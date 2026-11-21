"""Text normalization for ST: the ``st.sh`` case conventions.

ST recipes label each side of the corpus with a case convention and build a
separate text stream for it:

* ``tc``     truecased, punctuation kept (``egs2/must_c/st1`` target side)
* ``lc``     lowercased
* ``lc.rm``  lowercased with punctuation stripped (both sides of
  ``egs2/covost2/st1``, source side of ``egs2/must_c/st1``)

``local/data_prep.sh`` builds these with Moses' ``lowercase.perl`` and espnet's
``utils/remove_punctuation.pl``. Both are ported here so the recipes reproduce
the same character inventory instead of inventing their own normalization.

Moses' ``tokenizer.perl`` is deliberately NOT reproduced: it is tokenization,
which SentencePiece replaces.
"""

import string
import unicodedata

CASES = ("tc", "lc", "lc.rm")

# Perl's [[:punct:]] under Unicode rules matches \p{Punct} plus the ASCII
# symbols ($ + < = > ^ ` | ~), which Unicode classifies as Symbol.
_ASCII_PUNCT = frozenset(string.punctuation)


def is_punctuation(char: str) -> bool:
    """Return whether ``char`` is what Perl's ``[[:punct:]]`` would match.

    Examples:
        >>> is_punctuation("!")
        True
        >>> is_punctuation("$")  # Symbol to Unicode, punctuation to Perl
        True
        >>> is_punctuation("a")
        False
    """
    return unicodedata.category(char).startswith("P") or char in _ASCII_PUNCT


def remove_punctuation(text: str) -> str:
    """Port of espnet's ``utils/remove_punctuation.pl``.

    Apostrophes survive, and the ``<space>`` scoring marker is protected, both
    via the same placeholder trick the Perl script uses.

    Examples:
        >>> remove_punctuation("Don't stop, now!")
        "Don't stop now"
        >>> remove_punctuation("a <space> b.")
        'a <space> b'
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

    Examples:
        >>> apply_case("Mr. Smith's dog.", "tc")
        "Mr. Smith's dog."
        >>> apply_case("Mr. Smith's dog.", "lc")
        "mr. smith's dog."
        >>> apply_case("Mr. Smith's dog.", "lc.rm")
        "mr smith's dog"
    """
    if case == "tc":
        return text
    if case == "lc":
        return text.lower()
    if case == "lc.rm":
        return remove_punctuation(text.lower())
    raise ValueError(f"Unknown case {case!r}; expected one of {CASES}")
