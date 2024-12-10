import copy
import os
from typing import Iterable, List, Optional

from typeguard import typechecked

from espnet2.text.abs_tokenizer import AbsTokenizer

LANGUAGES_CODE_MAPPING = {
    "noinfo": "english",  # default, English
    "ca": "catalan",
    "cs": "czech",
    "cy": "welsh",
    "de": "german",
    "en": "english",
    "eu": "basque",
    "es": "spanish",
    "fa": "persian",
    "fr": "french",
    "it": "italian",
    "ja": "japanese",
    "jpn": "japanese",
    "ko": "korean",
    "kr": "korean",
    "nl": "dutch",
    "pl": "polish",
    "pt": "portuguese",
    "ru": "russian",
    "tt": "tatar",
    "zh": "chinese",
    "zh-TW": "chinese",
    "zh-CN": "chinese",
    "zh-HK": "chinese",
}
dirname = os.path.dirname(__file__)


class OpenAIWhisperTokenizer(AbsTokenizer):
    """
    A tokenizer for the OpenAI Whisper model.

    This class handles the tokenization process for both transcribing
    and translating text using the OpenAI Whisper model. It supports
    various languages and can utilize additional tokens if specified.

    Attributes:
        model (str): The model type of the Whisper tokenizer.
        language (str): The language code for the tokenizer.
        task (str): The task to perform, either 'transcribe' or 'translate'.
        tokenizer: The initialized tokenizer from the Whisper library.

    Args:
        model_type (str): The type of the Whisper model to use. Should be
            either "whisper_en" or "whisper_multilingual".
        language (str): The language code to use. Defaults to "en".
        task (str): The task to perform. Can be either "transcribe" or
            "translate". Defaults to "transcribe".
        sot (bool): A flag indicating whether to include start-of-token
            symbols. Defaults to False.
        speaker_change_symbol (str): The symbol to use for speaker changes.
            Defaults to "<sc>".
        added_tokens_txt (Optional[str]): A path to a text file containing
            additional tokens to be added to the tokenizer.

    Raises:
        ValueError: If the specified language or task is unsupported
            for the Whisper model.

    Examples:
        >>> tokenizer = OpenAIWhisperTokenizer(
        ...     model_type="whisper_multilingual",
        ...     language="fr",
        ...     task="transcribe"
        ... )
        >>> tokens = tokenizer.text2tokens("Bonjour, comment ça va?")
        >>> text = tokenizer.tokens2text(tokens)

    Note:
        Ensure that the Whisper library is properly installed. If the
        library is not found, an error message will be printed, and
        an exception will be raised.
    """

    @typechecked
    def __init__(
        self,
        model_type: str,
        language: str = "en",
        task: str = "transcribe",
        sot: bool = False,
        speaker_change_symbol: str = "<sc>",
        added_tokens_txt: Optional[str] = None,
    ):

        try:
            import whisper.tokenizer
        except Exception as e:
            print("Error: whisper is not properly installed.")
            print(
                "Please install whisper with: cd ${MAIN_ROOT}/tools && "
                "./installers/install_whisper.sh"
            )
            raise e

        self.model = model_type

        self.language = LANGUAGES_CODE_MAPPING.get(language)
        if self.language is None:
            raise ValueError(f"language: {self.language} unsupported for Whisper model")
        self.task = task
        if self.task not in ["transcribe", "translate"]:
            raise ValueError(f"task: {self.task} unsupported for Whisper model")

        if model_type == "whisper_en":
            self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False)
        elif model_type == "whisper_multilingual":
            self.tokenizer = whisper.tokenizer.get_tokenizer(
                multilingual=True, language=self.language, task=self.task
            )
            if added_tokens_txt is not None:
                _added_tokens = []
                with open(added_tokens_txt) as f:
                    lines = f.readlines()
                    for line in lines:
                        _added_tokens.append(line.rstrip())
                self.tokenizer.tokenizer.add_tokens(_added_tokens)
        else:
            raise ValueError("tokenizer unsupported:", model_type)

        self.tokenizer = copy.deepcopy(self.tokenizer)
        # Whisper uses discrete tokens (20ms) to encode timestamp
        timestamps = [f"<|{i*0.02:.2f}|>" for i in range(0, 1501)]
        sc = [speaker_change_symbol] if sot else []
        special_tokens = (
            self.tokenizer.tokenizer.additional_special_tokens + timestamps + sc
        )
        self.tokenizer.tokenizer.add_special_tokens(
            dict(additional_special_tokens=special_tokens)
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(model_type={self.model}, "
            f"language={self.language})"
        )

    def text2tokens(self, line: str) -> List[str]:
        """
            Convert a text line into a list of tokens.

        This method utilizes the underlying tokenizer to tokenize the provided
        text line. It does not add any special tokens during the tokenization
        process.

        Args:
            line (str): The input text line to be tokenized.

        Returns:
            List[str]: A list of tokens generated from the input text line.

        Examples:
            >>> tokenizer = OpenAIWhisperTokenizer(model_type="whisper_en")
            >>> tokens = tokenizer.text2tokens("Hello, how are you?")
            >>> print(tokens)
            ['Hello', ',', 'how', 'are', 'you', '?']

        Note:
            This method assumes that the tokenizer has been properly initialized
            and is ready for use.
        """
        return self.tokenizer.tokenizer.tokenize(line, add_special_tokens=False)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        """
            A tokenizer for OpenAI's Whisper model.

        This tokenizer is responsible for converting text to tokens and vice versa,
        tailored for the specific requirements of the Whisper model.

        Attributes:
            model (str): The type of Whisper model being used.
            language (str): The language for tokenization, mapped from a code.
            task (str): The task for which the model is used (transcribe/translate).
            tokenizer: The actual tokenizer instance used for token conversion.

        Args:
            model_type (str): The model type, either "whisper_en" or
                "whisper_multilingual".
            language (str, optional): The language code for tokenization. Defaults to "en".
            task (str, optional): The task to perform with the model. Can be "transcribe"
                or "translate". Defaults to "transcribe".
            sot (bool, optional): Whether to include start of transcription token.
                Defaults to False.
            speaker_change_symbol (str, optional): Symbol to denote speaker changes.
                Defaults to "<sc>".
            added_tokens_txt (Optional[str], optional): Path to a text file containing
                additional tokens to add. Defaults to None.

        Raises:
            ValueError: If an unsupported language or task is specified or if the
                tokenizer model type is unsupported.

        Examples:
            >>> tokenizer = OpenAIWhisperTokenizer(model_type="whisper_en")
            >>> tokens = tokenizer.text2tokens("Hello, world!")
            >>> text = tokenizer.tokens2text(tokens)
            >>> print(text)  # Output: "Hello, world!"

        Note:
            Make sure to have the Whisper package installed properly. If not, an
            error will be raised during initialization.
        """
        return self.tokenizer.tokenizer.convert_tokens_to_string(tokens)
