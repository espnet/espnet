from pathlib import Path
from typing import Dict, Iterable, List, Union

import sentencepiece as spm
from typeguard import typechecked

from espnet2.text.abs_tokenizer import AbsTokenizer


class SentencepiecesTokenizer(AbsTokenizer):
    """
        Tokenizer that utilizes SentencePiece for tokenization and detokenization.

    This class inherits from `AbsTokenizer` and provides methods to convert text
    to tokens and vice versa using the SentencePiece model specified during
    initialization. It lazily loads the SentencePiece processor to avoid issues
    with pickling, which can occur when using multiprocessing.

    Attributes:
        model (str): The path to the SentencePiece model file.
        encode_kwargs (Dict): Additional keyword arguments for encoding.

    Args:
        model (Union[Path, str]): The path to the SentencePiece model file.
        encode_kwargs (Dict, optional): Additional keyword arguments for the
            `EncodeAsPieces` method. Defaults to an empty dictionary.

    Examples:
        >>> tokenizer = SentencepiecesTokenizer("path/to/model.model")
        >>> tokens = tokenizer.text2tokens("Hello, world!")
        >>> print(tokens)
        ['Hello', ',', '▁world', '!']
        >>> text = tokenizer.tokens2text(tokens)
        >>> print(text)
        'Hello, world!'

    Raises:
        ValueError: If the model file does not exist or cannot be loaded.

    Note:
        The SentencePiece model must be trained and available at the specified
        path. Ensure that the model is compatible with the expected tokenization
        strategy.

    Todo:
        - Implement support for custom tokenization strategies.
    """

    @typechecked
    def __init__(self, model: Union[Path, str], encode_kwargs: Dict = dict()):
        self.model = str(model)
        # NOTE(kamo):
        # Don't build SentencePieceProcessor in __init__()
        # because it's not picklable and it may cause following error,
        # "TypeError: can't pickle SwigPyObject objects",
        # when giving it as argument of "multiprocessing.Process()".
        self.sp = None
        self.encode_kwargs = encode_kwargs

    def __repr__(self):
        return f'{self.__class__.__name__}(model="{self.model}")'

    def _build_sentence_piece_processor(self):
        # Build SentencePieceProcessor lazily.
        if self.sp is None:
            self.sp = spm.SentencePieceProcessor()
            self.sp.load(self.model)

    def text2tokens(self, line: str) -> List[str]:
        """
                Converts a given text line into a list of tokens using the SentencePiece model.

        This method utilizes the SentencePieceProcessor to encode the input text line
        into a sequence of tokens (pieces). It is important to ensure that the SentencePiece
        model is properly loaded before calling this method, which is handled internally.

        Args:
            line (str): The input text line that needs to be tokenized.

        Returns:
            List[str]: A list of tokens (pieces) generated from the input text line.

        Examples:
            tokenizer = SentencepiecesTokenizer(model="path/to/model")
            tokens = tokenizer.text2tokens("This is an example sentence.")
            print(tokens)  # Output might look like: ['This', 'is', 'an', 'example', 'sentence', '.']

        Note:
            Ensure that the SentencePiece model file exists and is accessible at the
            specified path during initialization of the tokenizer.

        Raises:
            Exception: Raises an exception if the SentencePiece model cannot be loaded.
        """
        self._build_sentence_piece_processor()
        return self.sp.EncodeAsPieces(line, **self.encode_kwargs)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        """
                Converts a sequence of tokens back into text using the SentencePiece model.

        This method requires that the SentencePieceProcessor is built, which is done
        lazily when this method is called. It takes an iterable of tokens and
        decodes them into a single string.

        Args:
            tokens (Iterable[str]): An iterable containing the tokens to be decoded.

        Returns:
            str: The decoded text corresponding to the input tokens.

        Examples:
            >>> tokenizer = SentencepiecesTokenizer("model_file.model")
            >>> tokens = ["▁Hello", "▁world", "!"]
            >>> text = tokenizer.tokens2text(tokens)
            >>> print(text)
            "Hello world!"

        Note:
            Ensure that the SentencePiece model is properly loaded before calling
            this method, as it relies on the model to decode the tokens.

        Raises:
            ValueError: If the input tokens are invalid or cannot be decoded.
        """
        self._build_sentence_piece_processor()
        return self.sp.DecodePieces(list(tokens))
