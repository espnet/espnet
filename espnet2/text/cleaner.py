from typing import Collection

from jaconv import jaconv
import tacotron_cleaner.cleaners
from typeguard import check_argument_types
from vietnamese_cleaner import vietnamese_cleaners


class TextCleaner:
    """Text cleaner

    Examples:
        >>> cleaner = TextCleaner("tacotron")
        >>> cleaner("(Hello-World);   &  jr. & dr.")
        'HELLO WORLD, AND JUNIOR AND DOCTOR'

    """

    def __init__(self, normalize_types: Collection[str] = None):
        assert check_argument_types()

        if normalize_types is None:
            self.normalize_types = []
        elif isinstance(normalize_types, str):
            self.normalize_types = [normalize_types]
        else:
            self.normalize_types = list(normalize_types)

    def __call__(self, text: str) -> str:
        for t in self.normalize_types:
            if t == "tacotron":
                text = tacotron_cleaner.cleaners.custom_english_cleaners(text)
            elif t == "jaconv":
                text = jaconv.normalize(text)
            elif t == "vietnamese":
                text = vietnamese_cleaners.vietnamese_cleaner(text)
            else:
                raise RuntimeError(f"Not supported: type={t}")

        return text
