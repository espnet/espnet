import copy
from typing import Iterable, List, Union

import numpy as np
from typeguard import check_argument_types

# <sos> and <eos> for Whisper multilingual ---
# '<|startoftranscript|>': 50258
# '<|endoftext|>':         50257

# <sos> and <eos> for Whisper english ---
# '<|startoftranscript|>': 50257
# '<|endoftext|>':         50256


class OpenAIWhisperTokenIDConverter:
    def __init__(
        self,
        model_type: str = "whisper_multilingual",
        sot: bool = False,
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

        if model_type == "whisper_en":
            self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False)
        # TODO(Shih-Lun): should support feeding in
        #                  different languages (default is en)
        elif model_type == "whisper_multilingual":
            self.tokenizer = whisper.tokenizer.get_tokenizer(
                multilingual=True, language=None
            )
        else:
            raise ValueError("tokenizer unsupported:", model_type)
        
        self.tokenizer = copy.deepcopy(self.tokenizer)
        timestamps = [f'<|{i*30/1500:.2f}|>' for i in range(0, 1501)]
        sc = ['<sc>'] if sot else []
        special_tokens = self.tokenizer.tokenizer.additional_special_tokens + timestamps + sc
        self.tokenizer.tokenizer.add_special_tokens(
            dict(additional_special_tokens=special_tokens)
        )
                

    def get_num_vocabulary_size(self) -> int:
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
