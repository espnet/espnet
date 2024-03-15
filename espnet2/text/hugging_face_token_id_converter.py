from typing import Iterable, List, Union

import numpy as np
from typeguard import typechecked

try:
    from transformers import AutoTokenizer

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


class HuggingFaceTokenIDConverter:
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
        return self.tokenizer.vocab_size

    def ids2tokens(self, integers: Union[np.ndarray, Iterable[int]]) -> List[str]:
        return self.tokenizer.convert_ids_to_tokens(integers)

    def tokens2ids(self, tokens: Iterable[str]) -> List[int]:
        return self.tokenizer.convert_tokens_to_ids(tokens)
