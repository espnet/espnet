import copy
import os
from typing import Iterable, List

from typeguard import check_argument_types

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
    def __init__(
        self,
        model_type: str,
        language: str = "en",
        task: str = "transcribe",
        sot: bool = False,
        speaker_change_symbol: str = "<sc>",
        added_tokens_txt: str = None,
    ):
        assert check_argument_types()

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
        return self.tokenizer.tokenizer.tokenize(line, add_special_tokens=False)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        return self.tokenizer.tokenizer.convert_tokens_to_string(tokens)
