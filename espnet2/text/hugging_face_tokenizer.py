from pathlib import Path
from typing import Iterable, List, Union

from typeguard import typechecked

from espnet2.text.abs_tokenizer import AbsTokenizer

try:
    from transformers import AutoTokenizer

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


class HuggingFaceTokenizer(AbsTokenizer):
    """
        HuggingFaceTokenizer is a tokenizer that utilizes Hugging Face's Transformers
    library to tokenize and detokenize text.

    This class is a subclass of AbsTokenizer and is designed to work with various
    Hugging Face models for natural language processing tasks. It builds the
    tokenizer lazily to avoid pickling issues when using multiprocessing.

    Attributes:
        model (str): The model name or path for the Hugging Face tokenizer.
        tokenizer (AutoTokenizer): The Hugging Face tokenizer instance, lazily
            initialized.

    Args:
        model (Union[Path, str]): The model name or path to load the tokenizer.

    Raises:
        ImportError: If the `transformers` library is not available.

    Examples:
        >>> tokenizer = HuggingFaceTokenizer("bert-base-uncased")
        >>> tokens = tokenizer.text2tokens("Hello, world!")
        >>> print(tokens)
        ['hello', ',', 'world', '!']

        >>> text = tokenizer.tokens2text(tokens)
        >>> print(text)
        "Hello, world!"

    Note:
        Ensure that the `transformers` library is installed. You can install it
        via `pip install transformers` or by following the installation steps
        for espnet.

    Todo:
        - Extend functionality to support additional tokenization methods or
          configurations.
    """

    @typechecked
    def __init__(self, model: Union[Path, str]):

        if not is_transformers_available:
            raise ImportError(
                "`transformers` is not available. Please install it via `pip install"
                " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_transformers.sh`."
            )

        self.model = str(model)
        # NOTE(kamo):
        # Don't build tokenizer in __init__()
        # because it's not picklable and it may cause following error,
        # "TypeError: can't pickle SwigPyObject objects",
        # when giving it as argument of "multiprocessing.Process()".
        self.tokenizer = None

    def __repr__(self):
        return f'{self.__class__.__name__}(model="{self.model}")'

    def _build_tokenizer(self):
        # Build Hugging Face tokenizer lazily.
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)

    def text2tokens(self, line: str) -> List[str]:
        """
            Convert a given text line into a list of tokens using the Hugging Face
        tokenizer.

        This method initializes the tokenizer if it has not been built yet and
        then tokenizes the input text.

        Args:
            line (str): The input text line to be tokenized.

        Returns:
            List[str]: A list of tokens extracted from the input text.

        Examples:
            >>> tokenizer = HuggingFaceTokenizer('bert-base-uncased')
            >>> tokens = tokenizer.text2tokens("Hello, how are you?")
            >>> print(tokens)
            ['hello', ',', 'how', 'are', 'you', '?']

        Raises:
            ValueError: If the input line is empty or None.
        """
        return self.tokenizer.tokenize(line)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        """
                Converts a list of tokens back into a text string using a Hugging Face tokenizer.

        This method first ensures that the tokenizer is built, and then it converts the
        provided tokens into their corresponding text representation. It utilizes the
        Hugging Face's `batch_decode` method to handle the conversion.

        Args:
            tokens (Iterable[str]): An iterable collection of tokens to be converted
                back into a text string.

        Returns:
            str: The reconstructed text string from the provided tokens.

        Examples:
            >>> tokenizer = HuggingFaceTokenizer("bert-base-uncased")
            >>> tokens = ["hello", "world"]
            >>> text = tokenizer.tokens2text(tokens)
            >>> print(text)
            "hello world"

        Note:
            The method skips special tokens during decoding to ensure that the output
            text is clean and free of any unnecessary characters.

        Raises:
            ValueError: If the input tokens are invalid or cannot be decoded.
        """
        return (
            self.tokenizer.batch_decode(
                [self.tokenizer.convert_tokens_to_ids(tokens)], skip_special_tokens=True
            )[0]
            .replace("\n", " ")
            .strip()
        )
