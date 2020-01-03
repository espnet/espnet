from pathlib import Path
from typing import Iterable
from typing import List
from typing import Union

import sentencepiece as spm
from typeguard import check_argument_types

from espnet2.text.abs_tokenizer import AbsTokenizer


class SentencepiecesTokenizer(AbsTokenizer):
    def __init__(self, model: Union[Path, str]):
        assert check_argument_types()
        self.model = Path(model)
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(str(model))

    def __repr__(self):
        return f'{self.__class__.__name__}(model="{self.model}")'

    def text2tokens(self, line: str) -> List[str]:
        return self.sp.EncodeAsPieces(line)

    def tokens2text(self, tokens: Iterable[str]) -> str:
        return self.sp.DecodePieces(list(tokens))
