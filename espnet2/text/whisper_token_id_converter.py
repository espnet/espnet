import copy
from typing import Iterable, List, Union

import numpy as np
from typeguard import check_argument_types

from espnet2.text.whisper_tokenizer import LANGUAGES_CODE_MAPPING

# <sos> and <eos> for Whisper multilingual ---
# '<|startoftranscript|>': 50258
# '<|endoftext|>':         50257

# <sos> and <eos> for Whisper english ---
# '<|startoftranscript|>': 50257
# '<|endoftext|>':         50256


class OpenAIWhisperTokenIDConverter:
    def __init__(
        self,
        model_type: str,
        language: str = "en",
        task: str = "transcribe",
        sot: bool = False,
        speaker_change_symbol: str = "<sc>",
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

        language = LANGUAGES_CODE_MAPPING.get(language)
        if language is None:
            raise ValueError(f"language: {language} unsupported for Whisper model")
        if task not in ["transcribe", "translate"]:
            raise ValueError(f"task: {task} unsupported for Whisper model")

        if model_type == "whisper_en":
            self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False)
        elif model_type == "whisper_multilingual":
            self.tokenizer = whisper.tokenizer.get_tokenizer(
                multilingual=True, language=language, task=task
            )
        else:
            raise ValueError("tokenizer unsupported:", model_type)

        self.tokenizer = copy.deepcopy(self.tokenizer)
        timestamps = [f"<|{i*30/1500:.2f}|>" for i in range(0, 1501)]
        sc = [speaker_change_symbol] if sot else []
        special_tokens = (
            self.tokenizer.tokenizer.additional_special_tokens + timestamps + sc
        )
        self.tokenizer.tokenizer.add_special_tokens(
            dict(additional_special_tokens=special_tokens)
        )
        self.model_type = model_type

    def get_num_vocabulary_size(self) -> int:
        if self.model_type == "whisper_en":
            return (
                self.tokenizer.tokenizer.vocab_size
                + len(self.tokenizer.tokenizer.get_added_vocab())
                - 1
            )
        return self.tokenizer.tokenizer.vocab_size + len(
            self.tokenizer.tokenizer.get_added_vocab()
        )

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        return self.tokenizer.tokenizer.convert_ids_to_tokens(
            integers, skip_special_tokens=True
        )

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        return list(
            self.tokenizer.sot_sequence_including_notimestamps[1:]
        ) + self.tokenizer.tokenizer.convert_tokens_to_ids(tokens)
