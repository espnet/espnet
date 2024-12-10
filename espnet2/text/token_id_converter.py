from pathlib import Path
from typing import Dict, Iterable, List, Union

import numpy as np
from typeguard import typechecked


class TokenIDConverter:
    """
        A class to convert between tokens and their corresponding IDs.

    This class facilitates the conversion of tokens to their integer IDs and vice versa.
    It takes a list of tokens, which can be provided as a file path, a string, or an
    iterable. It also allows for the specification of an unknown symbol to handle tokens
    that are not present in the provided token list.

    Attributes:
        token_list_repr (str): A string representation of the token list, showing the
            first few tokens and the total vocabulary size.
        token_list (List[str]): A list of tokens.
        token2id (Dict[str, int]): A dictionary mapping tokens to their corresponding
            integer IDs.
        unk_symbol (str): The symbol used for unknown tokens.
        unk_id (int): The integer ID corresponding to the unknown symbol.

    Args:
        token_list (Union[Path, str, Iterable[str]]): A list of tokens provided as a
            file path, string, or iterable.
        unk_symbol (str): The symbol to represent unknown tokens. Defaults to "<unk>".

    Raises:
        RuntimeError: If a duplicate token is found in the token list or if the
            unknown symbol does not exist in the token list.

    Examples:
        # Using a file containing tokens
        converter = TokenIDConverter("path/to/token_list.txt")

        # Using a list of tokens
        converter = TokenIDConverter(["hello", "world", "<unk>"])

        # Getting the vocabulary size
        vocab_size = converter.get_num_vocabulary_size()

        # Converting IDs to tokens
        tokens = converter.ids2tokens(np.array([0, 1, 2]))

        # Converting tokens to IDs
        ids = converter.tokens2ids(["hello", "unknown_token"])
    """

    @typechecked
    def __init__(
        self,
        token_list: Union[Path, str, Iterable[str]],
        unk_symbol: str = "<unk>",
    ):

        if isinstance(token_list, (Path, str)):
            token_list = Path(token_list)
            self.token_list_repr = str(token_list)
            self.token_list: List[str] = []

            with token_list.open("r", encoding="utf-8") as f:
                for idx, line in enumerate(f):
                    line = line[0] + line[1:].rstrip()
                    self.token_list.append(line)

        else:
            self.token_list: List[str] = list(token_list)
            self.token_list_repr = ""
            for i, t in enumerate(self.token_list):
                if i == 3:
                    break
                self.token_list_repr += f"{t}, "
            self.token_list_repr += f"... (NVocab={(len(self.token_list))})"

        self.token2id: Dict[str, int] = {}
        for i, t in enumerate(self.token_list):
            if t in self.token2id:
                raise RuntimeError(f'Symbol "{t}" is duplicated')
            self.token2id[t] = i

        self.unk_symbol = unk_symbol
        if self.unk_symbol not in self.token2id:
            raise RuntimeError(
                f"Unknown symbol '{unk_symbol}' doesn't exist in the token_list"
            )
        self.unk_id = self.token2id[self.unk_symbol]

    def get_num_vocabulary_size(self) -> int:
        """
                Retrieves the size of the vocabulary, which is the number of unique tokens.

        This method returns the total number of tokens stored in the `token_list`
        attribute of the `TokenIDConverter` class. It is useful for understanding
        the vocabulary size that the converter can work with.

        Returns:
            int: The number of unique tokens in the vocabulary.

        Examples:
            >>> converter = TokenIDConverter(["hello", "world", "<unk>"])
            >>> converter.get_num_vocabulary_size()
            3

        Note:
            This method counts the number of tokens as they are stored in the
            `token_list` attribute. It does not account for any potential
            duplicates, as duplicates are not allowed during initialization.
        """
        return len(self.token_list)

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        """
                Converts a list of token IDs (integers) back into their corresponding tokens
        (string representations) based on a predefined token list.

        This method can handle both NumPy arrays and iterable collections of integers.
        If an integer does not correspond to any token in the token list, it will be
        skipped.

        Args:
            integers (Union[np.ndarray, Iterable[int]]): A 1-dimensional array or
                iterable containing the integer token IDs to convert to tokens.

        Returns:
            List[str]: A list of tokens corresponding to the provided integer IDs.

        Raises:
            ValueError: If the input `integers` is a NumPy array that is not
                1-dimensional.

        Examples:
            >>> converter = TokenIDConverter(['hello', 'world', '<unk>'])
            >>> converter.ids2tokens([0, 1, 2])
            ['hello', 'world', '<unk>']

            >>> converter.ids2tokens(np.array([0, 2]))
            ['hello', '<unk>']

            >>> converter.ids2tokens(np.array([[0, 1]]))  # This will raise ValueError
        """
        if isinstance(integers, np.ndarray) and integers.ndim != 1:
            raise ValueError(f"Must be 1 dim ndarray, but got {integers.ndim}")
        return [self.token_list[i] for i in integers]

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        """
                Converts tokens to their corresponding IDs using a predefined token list.

        This method retrieves the ID for each token provided in the input iterable. If a
        token is not found in the token-to-ID mapping, it returns the ID for the unknown
        symbol.

        Args:
            tokens (Iterable[str]): An iterable of tokens for which to retrieve IDs.

        Returns:
            List[int]: A list of corresponding IDs for the input tokens. If a token
            is not found, the ID for the unknown symbol is used.

        Examples:
            >>> converter = TokenIDConverter(["hello", "world", "<unk>"])
            >>> converter.tokens2ids(["hello", "world", "unknown_token"])
            [0, 1, 2]

            >>> converter.tokens2ids(["hello", "<unk>"])
            [0, 2]

        Note:
            The unknown symbol must be part of the initial token list; otherwise, a
            RuntimeError will be raised during the initialization of the
            TokenIDConverter class.
        """
        return [self.token2id.get(i, self.unk_id) for i in tokens]
