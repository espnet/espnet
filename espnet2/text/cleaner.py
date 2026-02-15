from typing import Collection, Optional

from typeguard import typechecked

try:
    import tacotron_cleaner.cleaners as tacotron_cleaners
    from jaconv import jaconv
except ImportError:
    tacotron_cleaners = None
    jaconv = None

# We removed underthesea from tts extra requirement because it
# causes installation issues with pyproject.toml (See #6239 for details).
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
    """Text cleaner.

    Examples:
        >>> cleaner = TextCleaner("tacotron")
        >>> cleaner("(Hello-World);   &  jr. & dr.")
        'HELLO WORLD, AND JUNIOR AND DOCTOR'

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
                if tacotron_cleaners is None:
                    raise RuntimeError(
                        "Please install espnet with `pip install espnet[tts]`"
                    )
                text = tacotron_cleaners.custom_english_cleaners(text)
            elif t == "jaconv":
                if jaconv is None:
                    raise RuntimeError(
                        "Please install espnet with `pip install espnet[tts]`"
                    )
                text = jaconv.normalize(text)
            elif t == "vietnamese":
                if vietnamese_cleaners is None:
                    raise RuntimeError(
                        "Please install underthesea" "by `pip install underthesea`"
                    )
                text = vietnamese_cleaners.vietnamese_cleaner(text)
            elif t == "korean_cleaner":
                text = KoreanCleaner.normalize_text(text)
            elif "whisper" in t and self.whisper_cleaner is not None:
                text = self.whisper_cleaner(text)
            else:
                raise RuntimeError(f"Not supported: type={t}")

        return text
