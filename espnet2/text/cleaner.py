from typing import Collection

from jaconv import jaconv
from tacotron_cleaner.cleaners import custom_english_cleaners
from typeguard import check_argument_types

try:
    from vietnamese_cleaner import vietnamese_cleaners
except ImportError:
    vietnamese_cleaners = None

from espnet2.text.korean_cleaner import KoreanCleaner
from espnet2.text.mfa_cleaners import mfa_english_cleaner

try:
    from whisper.normalizers import BasicTextNormalizer, EnglishTextNormalizer
except (ImportError, SyntaxError):
    BasicTextNormalizer = None


class TextCleaner:
    """Text cleaner.

    Examples:
        >>> cleaner = TextCleaner("tacotron")
        >>> cleaner("(Hello-World);   &  jr. & dr.")
        'HELLO WORLD, AND JUNIOR AND DOCTOR'

    """

    def __init__(self, cleaner_types: Collection[str] = None):
        assert check_argument_types()

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

        self.cleaner_fns = []
        for t in self.cleaner_types:
            if t == "tacotron":
                cleaner_fn = custom_english_cleaners
            elif t == "mfa_english":
                cleaner_fn = mfa_english_cleaner
            elif t == "jaconv":
                cleaner_fn = jaconv.normalize
            elif t == "vietnamese":
                if vietnamese_cleaners is None:
                    raise RuntimeError("Please install underthesea")
                cleaner_fn = vietnamese_cleaners.vietnamese_cleaner
            elif t == "korean_cleaner":
                cleaner_fn = KoreanCleaner.normalize_text
            elif "whisper" in t and self.whisper_cleaner is not None:
                cleaner_fn = self.whisper_cleaner
            else:
                raise RuntimeError(f"Not supported: type={t}")
            self.cleaner_fns.append(cleaner_fn)

    def __call__(self, text: str) -> str:
        for cleaner_fn in self.cleaner_fns:
            text = cleaner_fn(text)
        return text
