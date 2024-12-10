import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Union

from typeguard import typechecked

from espnet2.text.abs_tokenizer import AbsTokenizer


class WordTokenizer(AbsTokenizer):
    """
        A tokenizer that splits text into words based on a specified delimiter and can
    optionally remove non-linguistic symbols.

    Attributes:
        delimiter (Optional[str]): The character used to split the text into tokens.
        non_linguistic_symbols (set): A set of symbols that are considered non-
            linguistic and can be removed from the tokenized output.
        remove_non_linguistic_symbols (bool): A flag indicating whether to remove
            non-linguistic symbols from the tokenized output.

    Args:
        delimiter (Optional[str]): The delimiter used for tokenization. If None,
            whitespace will be used as the default delimiter.
        non_linguistic_symbols (Union[Path, str, Iterable[str], None]): A path to a
            file or an iterable containing non-linguistic symbols to be removed. If
            None, no symbols will be removed.
        remove_non_linguistic_symbols (bool): If True, non-linguistic symbols will
            be removed from the tokenized output.

    Raises:
        Warning: If non_linguistic_symbols is provided while
            remove_non_linguistic_symbols is False.

    Examples:
        tokenizer = WordTokenizer(delimiter=",", non_linguistic_symbols=["#", "$"])
        tokens = tokenizer.text2tokens("Hello, world! # $")
        print(tokens)  # Output: ['Hello', ' world! ']

        text = tokenizer.tokens2text(tokens)
        print(text)  # Output: "Hello, world! "
    """

    @typechecked
    def __init__(
        self,
        delimiter: Optional[str] = None,
        non_linguistic_symbols: Union[Path, str, Iterable[str], None] = None,
        remove_non_linguistic_symbols: bool = False,
    ):
        self.delimiter = delimiter

        if not remove_non_linguistic_symbols and non_linguistic_symbols is not None:
            warnings.warn(
                "non_linguistic_symbols is only used "
                "when remove_non_linguistic_symbols = True"
            )

        if non_linguistic_symbols is None:
            self.non_linguistic_symbols = set()
        elif isinstance(non_linguistic_symbols, (Path, str)):
            non_linguistic_symbols = Path(non_linguistic_symbols)
            try:
                with non_linguistic_symbols.open("r", encoding="utf-8") as f:
                    self.non_linguistic_symbols = set(line.rstrip() for line in f)
            except FileNotFoundError:
                warnings.warn(f"{non_linguistic_symbols} doesn't exist.")
                self.non_linguistic_symbols = set()
        else:
            self.non_linguistic_symbols = set(non_linguistic_symbols)
        self.remove_non_linguistic_symbols = remove_non_linguistic_symbols

    def __repr__(self):
        return f'{self.__class__.__name__}(delimiter="{self.delimiter}")'

    def text2tokens(self, line: str) -> List[str]:
        """
                Converts a given text line into a list of tokens based on a specified delimiter.

        This method splits the input text `line` into tokens using the `delimiter` set
        during the initialization of the `WordTokenizer` instance. If the
        `remove_non_linguistic_symbols` attribute is set to `True`, any tokens that
        match the non-linguistic symbols will be excluded from the result.

        Args:
            line (str): The input text line to be tokenized.

        Returns:
            List[str]: A list of tokens extracted from the input text line.

        Examples:
            >>> tokenizer = WordTokenizer(delimiter=' ')
            >>> tokenizer.text2tokens("Hello world! This is a test.")
            ['Hello', 'world!', 'This', 'is', 'a', 'test.']

            >>> tokenizer = WordTokenizer(delimiter=',',
            ...                           non_linguistic_symbols=['n/a'],
            ...                           remove_non_linguistic_symbols=True)
            >>> tokenizer.text2tokens("value1,n/a,value2,value3")
            ['value1', 'value2', 'value3']
        """
        tokens = []
        for t in line.split(self.delimiter):
            if self.remove_non_linguistic_symbols and t in self.non_linguistic_symbols:
                continue
            tokens.append(t)
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        """
                Converts a list of tokens back into a text string using a specified delimiter.

        This method takes an iterable of tokens and joins them into a single string,
        inserting the specified delimiter between each token. If no delimiter has been
        set during the initialization of the WordTokenizer, a space character will be
        used as the default delimiter.

        Args:
            tokens (Iterable[str]): An iterable containing the tokens to be joined.

        Returns:
            str: A string representing the joined tokens, separated by the specified
            delimiter.

        Examples:
            >>> tokenizer = WordTokenizer(delimiter=", ")
            >>> tokens = ["Hello", "world", "!"]
            >>> text = tokenizer.tokens2text(tokens)
            >>> print(text)
            Hello, world, !

            >>> tokenizer_no_delimiter = WordTokenizer()
            >>> tokens_no_delimiter = ["Hello", "world", "!"]
            >>> text_no_delimiter = tokenizer_no_delimiter.tokens2text(tokens_no_delimiter)
            >>> print(text_no_delimiter)
            Hello world !
        """
        if self.delimiter is None:
            delimiter = " "
        else:
            delimiter = self.delimiter
        return delimiter.join(tokens)
