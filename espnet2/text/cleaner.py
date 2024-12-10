from typing import Collection, Optional

import tacotron_cleaner.cleaners
from jaconv import jaconv
from typeguard import typechecked

try:
    from vietnamese_cleaner import vietnamese_cleaners
except ImportError:
    vietnamese_cleaners = None

from espnet2.text.korean_cleaner import KoreanCleaner

try:
    from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
except (ImportError, SyntaxError):
    BasicTextNormalizer = None


class TextCleaner:
    """
        Text cleaner.

    This class provides various text cleaning functionalities based on specified
    cleaner types. It supports different cleaning methods including tacotron,
    jaconv, Vietnamese, Korean, and whisper text normalization.

    Attributes:
        cleaner_types (list): A list of cleaner types to be applied.

    Args:
        cleaner_types (Optional[Collection[str]]): A collection of cleaner types.
            If None, an empty list is used. It can also be a single string.

    Returns:
        str: The cleaned text after applying the specified cleaners.

    Raises:
        RuntimeError: If an unsupported cleaner type is specified or if the
        Vietnamese cleaner is requested but not available.

    Examples:
        >>> cleaner = TextCleaner("tacotron")
        >>> cleaner("(Hello-World);   &  jr. & dr.")
        'HELLO WORLD, AND JUNIOR AND DOCTOR'

    Note:
        Make sure to install required dependencies for all cleaner types to work
        properly, especially for Vietnamese cleaning.

    Todo:
        - Add more cleaner types and functionalities as needed.
    """

    @typechecked
    def __init__(self, cleaner_types: Optional[Collection[str]] = None):

        if cleaner_types is None:
            self.cleaner_types = []
        elif isinstance(cleaner_types, str):
            self.cleaner_types = [cleaner_types]
        else:
            self.cleaner_types = list(cleaner_types)

        self.whisper_cleaner = None
        if BasicTextNormalizer is not None:
            for t in self.cleaner_types:
                if t == "whisper_en":
                    self.whisper_cleaner = EnglishTextNormalizer()
                elif t == "whisper_basic":
                    self.whisper_cleaner = BasicTextNormalizer()

    def __call__(self, text: str) -> str:
        for t in self.cleaner_types:
            if t == "tacotron":
                text = tacotron_cleaner.cleaners.custom_english_cleaners(text)
            elif t == "jaconv":
                text = jaconv.normalize(text)
            elif t == "vietnamese":
                if vietnamese_cleaners is None:
                    raise RuntimeError("Please install underthesea")
                text = vietnamese_cleaners.vietnamese_cleaner(text)
            elif t == "korean_cleaner":
                text = KoreanCleaner.normalize_text(text)
            elif "whisper" in t and self.whisper_cleaner is not None:
                text = self.whisper_cleaner(text)
            else:
                raise RuntimeError(f"Not supported: type={t}")

        return text
