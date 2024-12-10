from typing import Iterable, List, Union

import numpy as np
from typeguard import typechecked

try:
    from transformers import AutoTokenizer

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


class HuggingFaceTokenIDConverter:
    """
        A converter class for transforming between token IDs and tokens using the
    Hugging Face Transformers library.

    This class provides methods to convert between token IDs and their corresponding
    tokens as well as to retrieve the vocabulary size of a specified model. It requires
    the `transformers` library to be installed.

    Attributes:
        tokenizer: An instance of `AutoTokenizer` initialized with the specified
        model.

    Args:
        model_name_or_path (str): The name or path of the pre-trained model to
        load the tokenizer from.

    Raises:
        ImportError: If the `transformers` library is not available.

    Examples:
        >>> converter = HuggingFaceTokenIDConverter('bert-base-uncased')
        >>> vocab_size = converter.get_num_vocabulary_size()
        >>> token_ids = converter.tokens2ids(['hello', 'world'])
        >>> tokens = converter.ids2tokens(token_ids)

    Note:
        Ensure that the model specified has a compatible tokenizer available.

    Todo:
        - Add support for custom tokenizers.
    """

    @typechecked
    def __init__(
        self,
        model_name_or_path: str,
    ):

        if not is_transformers_available:
            raise ImportError(
                "`transformers` is not available. Please install it via `pip install"
                " transformers` or `cd /path/to/espnet/tools && . ./activate_python.sh"
                " && ./installers/install_transformers.sh`."
            )

        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

    def get_num_vocabulary_size(self) -> int:
        """
                Returns the size of the vocabulary used by the tokenizer.

        This method accesses the `vocab_size` attribute of the tokenizer, which is
        initialized with a specified model. The vocabulary size indicates the total
        number of unique tokens that the tokenizer can recognize.

        Returns:
            int: The size of the vocabulary.

        Examples:
            >>> converter = HuggingFaceTokenIDConverter('bert-base-uncased')
            >>> vocab_size = converter.get_num_vocabulary_size()
            >>> print(vocab_size)
            30522  # Example size for BERT tokenizer
        """
        return self.tokenizer.vocab_size

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        """
                Converts a list of token IDs into their corresponding token strings using the
        Hugging Face tokenizer.

        This method is useful for translating numeric representations of tokens back
        into their readable string format, allowing for better interpretation of model
        outputs.

        Args:
            integers (Union[np.ndarray, Iterable[int]]): A collection of token IDs
                (integers) to be converted into tokens. This can be a NumPy array or
                any iterable containing integers.

        Returns:
            List[str]: A list of tokens corresponding to the input token IDs.

        Examples:
            >>> converter = HuggingFaceTokenIDConverter('bert-base-uncased')
            >>> token_ids = [101, 7592, 102]
            >>> tokens = converter.ids2tokens(token_ids)
            >>> print(tokens)
            ['[CLS]', 'hello', '[SEP]']

        Note:
            Ensure that the input integers are valid token IDs for the specified
            tokenizer. Invalid IDs may result in unexpected tokens or errors.
        """
        return self.tokenizer.convert_ids_to_tokens(integers)

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        """
                Converts tokens to their corresponding IDs using a Hugging Face tokenizer.

        This method takes an iterable of tokens (strings) and returns a list of
        integers representing the corresponding token IDs as defined by the
        Hugging Face tokenizer.

        Args:
            tokens (Iterable[str]): An iterable containing the tokens to be converted
                to IDs.

        Returns:
            List[int]: A list of integers representing the token IDs corresponding
                to the provided tokens.

        Examples:
            >>> converter = HuggingFaceTokenIDConverter("bert-base-uncased")
            >>> tokens = ["hello", "world"]
            >>> ids = converter.tokens2ids(tokens)
            >>> print(ids)  # Output: [7592, 2088] (IDs may vary based on the model)

        Note:
            Ensure that the tokens are valid for the tokenizer used. Invalid tokens
            may result in unexpected behavior or errors.
        """
        return self.tokenizer.convert_tokens_to_ids(tokens)
