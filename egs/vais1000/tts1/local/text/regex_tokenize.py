# -*- coding: utf-8 -*-
r"""This code inspired from underthesea package, edited by enamoria.

What changed from the original version: PSG. consider to be abbreviation, but not with the dot.
Just add a boundary for word at r"[A-ZĐ]+\s"
"""

import re
import sys

from underthesea.feature_engineering.text import Text

specials = [r"==>", r"->", r"\.\.\.", r">>", r"=\)\)"]
digit = r"\d+([\.,_]\d+)+"
email = r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+"

# urls pattern from nltk
# https://www.nltk.org/_modules/nltk/tokenize/casual.html
# with Vu Anh's modified to match fpt protocol
urls = r"""             # Capture 1: entire matched URL
  (?:
  (ftp|http)s?:               # URL protocol and colon
    (?:
      /{1,3}            # 1-3 slashes
      |                 #   or
      [a-z0-9%]         # Single letter or digit or '%'
                        # (Trying not to match e.g. "URI::Escape")
    )
    |                   #   or
                        # looks like domain name followed by a slash:
    [a-z0-9.\-]+[.]
    (?:[a-z]{2,13})
    /
  )
  (?:                                  # One or more:
    [^\s()<>{}\[\]]+                   # Run of non-space, non-()<>{}[]
    |                                  #   or
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)                        # balanced parens, non-recursive: (...)
  )+
  (?:                                  # End with:
    \([^\s()]*?\([^\s()]+\)[^\s()]*?\) # balanced parens, one level deep: (...(...)...)
    |
    \([^\s]+?\)                        # balanced parens, non-recursive: (...)
    |                                  #   or
    [^\s`!()\[\]{};:'".,<>?«»“”‘’]     # not a space or one of these punct chars
  )
  |                        # OR, the following to match naked domains:
  (?:
    (?<!@)                 # not preceded by a @, avoid matching foo@_gmail.com_
    [a-z0-9]+
    (?:[.\-][a-z0-9]+)*
    [.]
    (?:[a-z]{2,13})
    \b
    /?
    (?!@)                  # not succeeded by a @,
                           # avoid matching "foo.na" in "foo.na@example.com"
  )
"""
datetime = [
    r"\d{1,2}\/\d{1,2}(\/\d+)?",
    r"\d{1,2}-\d{1,2}(-\d+)?",
]
word = r"\w+"
non_word = r"[^\w\s]"
abbreviations = [
    r"[A-ZĐ]+\s",
    r"Tp\.",
    r"Mr\.", "Mrs\.", "Ms\.",
    r"Dr\.", "ThS\."
]
patterns = []
patterns.extend(abbreviations)
patterns.extend(specials)
patterns.extend([urls])
patterns.extend([email])
patterns.extend(datetime)
patterns.extend([digit])
patterns.extend([non_word])
patterns.extend([word])

patterns = "(" + "|".join(patterns) + ")"
if sys.version_info < (3, 0):
    patterns = patterns.decode('utf-8')
patterns = re.compile(patterns, re.VERBOSE | re.UNICODE)


def tokenize(text, format=None):
    """Tokenize text for word segmentation.

    :param text: raw text input
    :return: tokenize text
    """
    text = Text(text.lower())
    text = text.replace("\t", " ")
    tokens = re.findall(patterns, text)
    tokens = [token[0] for token in tokens]
    if format == "text":
        return " ".join(tokens)
    else:
        return tokens
