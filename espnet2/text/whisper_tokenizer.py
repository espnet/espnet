from typing import Iterable, List

from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer
import os

dirname = os.path.dirname(__file__)


class OpenAIWhisperTokenizer(AbsTokenizer):
    def __init__(self, model_type: str, added_tokens_txt: str = None):
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
        if model_type == "whisper_en":
            self.tokenizer = whisper.tokenizer.get_tokenizer(multilingual=False)
        # TODO(Shih-Lun): should support feeding in
        #                  different languages (default is en)
        elif model_type == "whisper_multilingual":
            self.tokenizer = whisper.tokenizer.get_tokenizer(
                multilingual=True, language=None
            )
            # import pdb;pdb.set_trace()
            if added_tokens_txt is not None:
                _added_tokens = []
                with open(added_tokens_txt) as f:
                    lines = f.readlines()
                    for line in lines:
                        _added_tokens.append(line.rstrip())
                self.tokenizer.tokenizer.add_tokens(_added_tokens)
        else:
            raise ValueError("tokenizer unsupported:", model_type)

    def __repr__(self):
        return f'{self.__class__.__name__}(model="{self.model}")'

    def text2tokens(self, line: str) -> List[str]:
        return self.tokenizer.tokenizer.tokenize(line, add_special_tokens=False)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        return self.tokenizer.tokenizer.convert_tokens_to_string(tokens)
