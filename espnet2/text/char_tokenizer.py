import warnings
from pathlib import Path
from typing import Iterable, List, Optional, Union

from typeguard import typechecked

from espnet2.text.abs_tokenizer import AbsTokenizer


class CharTokenizer(AbsTokenizer):
    """
        CharTokenizer is a character-level tokenizer that converts text to tokens and vice versa.

    This tokenizer handles non-linguistic symbols and allows customization of space
    representation. It can be used to preprocess text for various natural language
    processing tasks.

    Attributes:
        space_symbol (str): The symbol used to represent space in the tokenized output.
        non_linguistic_symbols (set): A set of non-linguistic symbols that will be
            treated as individual tokens.
        remove_non_linguistic_symbols (bool): If True, non-linguistic symbols will
            be removed from the tokenized output.
        nonsplit_symbols (set): A set of symbols that will not be split when tokenizing.

    Args:
        non_linguistic_symbols (Optional[Union[Path, str, Iterable[str]]]): A path
            to a file or a list of non-linguistic symbols to be treated as individual
            tokens. Defaults to None.
        space_symbol (str): The symbol used to represent space. Defaults to "<space>".
        remove_non_linguistic_symbols (bool): If True, removes non-linguistic symbols
            from the output. Defaults to False.
        nonsplit_symbols (Optional[Iterable[str]]): A list of symbols that should not
            be split when tokenizing. Defaults to None.

    Examples:
        tokenizer = CharTokenizer(non_linguistic_symbols=["#", "@"])
        tokens = tokenizer.text2tokens("Hello #World!")
        print(tokens)  # Output: ['H', 'e', 'l', 'l', 'o', ' ', '#', 'W', 'o', 'r', 'l', 'd', '!']
        text = tokenizer.tokens2text(tokens)
        print(text)  # Output: "Hello #World!"

    Raises:
        FileNotFoundError: If a specified file for non-linguistic symbols does not
            exist.

    Note:
        This tokenizer is part of the ESPnet2 text processing module.
    """

    @typechecked
    def __init__(
        self,
        non_linguistic_symbols: Optional[Union[Path, str, Iterable[str]]] = None,
        space_symbol: str = "<space>",
        remove_non_linguistic_symbols: bool = False,
        nonsplit_symbols: Optional[Iterable[str]] = None,
    ):
        self.space_symbol = space_symbol
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
        self.nonsplit_symbols = (
            set()
            if nonsplit_symbols is None
            else set([sym.split(":")[0] for sym in nonsplit_symbols])
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f'space_symbol="{self.space_symbol}"'
            f'non_linguistic_symbols="{self.non_linguistic_symbols}"'
            f'nonsplit_symbols="{self.nonsplit_symbols}"'
            f")"
        )

    def text2tokens(self, line: str) -> List[str]:
        """
                Converts a string of text into a list of tokens based on specified rules.

        The `text2tokens` method processes the input string `line` and returns a list of
        tokens. It recognizes both linguistic and non-linguistic symbols, as well as
        handles space characters according to the defined `space_symbol`.

        Args:
            line (str): The input string to be tokenized.

        Returns:
            List[str]: A list of tokens extracted from the input string.

        Examples:
            tokenizer = CharTokenizer(non_linguistic_symbols=["@", "#"], space_symbol="<space>")
            tokens = tokenizer.text2tokens("Hello @world! How are you?")
            print(tokens)  # Output: ['H', 'e', 'l', 'l', 'o', '<space>', '@', 'w', 'o', 'r', 'l', 'd', '!', '<space>', 'H', 'o', 'w', '<space>', 'a', 'r', 'e', '<space>', 'y', 'o', 'u', '?']

        Note:
            The behavior of this method is influenced by the `remove_non_linguistic_symbols`
            and `nonsplit_symbols` attributes set during the initialization of the
            `CharTokenizer` instance.
        """
        tokens = []
        while len(line) != 0:
            for w in self.non_linguistic_symbols.union(self.nonsplit_symbols):
                if line.startswith(w):
                    if (
                        w in self.nonsplit_symbols
                        or not self.remove_non_linguistic_symbols
                    ):
                        tokens.append(line[: len(w)])
                    line = line[len(w) :]
                    break
            else:
                t = line[0]
                if t == " ":
                    t = self.space_symbol
                tokens.append(t)
                line = line[1:]
        return tokens

    def tokens2text(self, tokens: Iterable[str]) -> str:
        """
                Converts a sequence of tokens back into a single text string.

        This method takes an iterable of tokens and transforms them into a single
        string representation. It replaces the specified space symbol with a space
        character to reconstruct the original text format.

        Args:
            tokens (Iterable[str]): An iterable containing tokens to be converted
                into text. The tokens can include a special space symbol that will
                be replaced with a regular space in the output.

        Returns:
            str: The reconstructed text string derived from the input tokens.

        Examples:
            >>> tokenizer = CharTokenizer()
            >>> tokens = ['H', 'e', 'l', 'l', 'o', '<space>', 'W', 'o', 'r', 'l', 'd']
            >>> tokenizer.tokens2text(tokens)
            'Hello World'

            >>> tokens = ['T', 'h', 'i', 's', '<space>', 'i', 's', '<space>', 'a',
            ... 'n', ' ', 'e', 'x', 'a', 'm', 'p', 'l', 'e', '.']
            >>> tokenizer.tokens2text(tokens)
            'This is an example.'
        """
        tokens = [t if t != self.space_symbol else " " for t in tokens]
        return "".join(tokens)
