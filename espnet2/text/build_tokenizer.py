from pathlib import Path
from typing import Iterable
from typing import Union

from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer
from espnet2.text.char_tokenizer import CharTokenizer
from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer
from espnet2.text.word_tokenizer import WordTokenizer


def build_tokenizer(
    token_type: str,
    bpemodel: Union[Path, str, Iterable[str]] = None,
    non_linguistic_symbols: Union[Path, str, Iterable[str]] = None,
    remove_non_linguistic_symbols: bool = False,
    space_symbol: str = "<space>",
    delimiter: str = None,
) -> AbsTokenizer:
    """A helper function to instantiate Tokenizer"""
    assert check_argument_types()
    if token_type == "bpe":
        if bpemodel is None:
            raise ValueError('bpemodel is required if token_type = "bpe"')

        if remove_non_linguistic_symbols:
            raise RuntimeError(
                "remove_non_linguistic_symbols is not implemented for token_type=bpe"
            )
        return SentencepiecesTokenizer(bpemodel)

    elif token_type == "word":
        if remove_non_linguistic_symbols and non_linguistic_symbols is not None:
            return WordTokenizer(
                delimiter=delimiter,
                non_linguistic_symbols=non_linguistic_symbols,
                remove_non_linguistic_symbols=True,
            )
        else:
            return WordTokenizer(delimiter=delimiter)

    elif token_type == "char":
        return CharTokenizer(
            non_linguistic_symbols=non_linguistic_symbols,
            space_symbol=space_symbol,
            remove_non_linguistic_symbols=remove_non_linguistic_symbols,
        )

    else:
        raise ValueError(
            f"token_mode must be one of bpe, word, or char: " f"{token_type}"
        )
