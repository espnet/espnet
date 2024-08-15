#!/usr/bin/env python3

# Copyright 2024 Jinchuan Tian
#  Apache 2.0  (http://www.apache.org/licenses/LICENSE-2.0)

import torch

from espnet2.speechlm.tokenizer.abs_tokenizer import AbsTokenizer
from espnet2.text.sentencepiece_tokenizer import SentencepiecesTokenizer


class TextBPETokenizer(AbsTokenizer):
    """A warpper for BPE tokenizer, only used for speechlm BPE detokenization"""

    def __init__(self, model, token_list):
        super(TextBPETokenizer, self).__init__()
        self.bpe = SentencepiecesTokenizer(model)
        self.token_list = token_list

    def forward(self, text):
        raise NotImplementedError

    def tokens2text(self, tokens):
        """
        tokens (torch.Tensor), token tensor, in size [B, T_text]
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.cpu().tolist()
        tokens = [
            self.bpe.tokens2text([self.token_list[tok] for tok in seq])
            for seq in tokens
        ]
        return tokens