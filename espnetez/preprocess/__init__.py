"""Tokenization utilities package.

This package provides a convenient flat namespace for sentence‑piece
tokenization and related helper functions.  The top‑level ``__init__``
re‑exports everything from :mod:`.sentencepiece` and :mod:`.tokenizer`,
so users can import the most common classes and functions directly
from the package without knowing the underlying module layout.

Typical usage
-------------

>>> from mypackage import SentencePieceTokenizer, tokenize
>>> tokenizer = SentencePieceTokenizer(model_path="spm.model")
>>> tokens = tokenize("Hello world")
>>> print(tokens)
['▁Hello', '▁world']

The re‑exported names include:

* :class:`.sentencepiece.SentencePieceTokenizer`
* :func:`.tokenizer.tokenize`
* :func:`.tokenizer.detokenize`
* etc.

See the individual modules for detailed documentation of each
class and function.  The package is intended to maintain backward
compatibility with earlier releases while simplifying the import
process for end users.
"""

from .sentencepiece import *
from .tokenizer import *
