from typing import Any, List, Tuple

import torch
import torch.nn as nn

from espnet2.lm.abs_model import AbsLM
from espnet.nets.pytorch_backend.transformer.embedding import PositionalEncoding
from espnet.nets.pytorch_backend.transformer.encoder import Encoder
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask


try:
    from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM, AutoTokenizer
    from transformers.file_utils import ModelOutput

    is_transformers_available = True
except ImportError:
    is_transformers_available = False


class TransformerLMOPT(AbsLM):
    def __init__(
        self,
        vocab_size: int,
        opt_name: str,
    ):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(opt_name, torch_dtype=torch.float16)
        self.vocab_size = vocab_size

        self.model.resize_token_embeddings(vocab_size)


    def _target_mask(self, ys_in_pad):
        ys_mask = ys_in_pad != 0
        m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
        return ys_mask.unsqueeze(-2) & m

    def forward(self, input: torch.Tensor, hidden: None) -> Tuple[torch.Tensor, None]:
        """Compute LM loss value from buffer sequences.

        Args:
            input (torch.Tensor): Input ids. (batch, len)
            hidden (torch.Tensor): Target ids. (batch, len)

        """
        output = self.model(input, labels=hidden)
        
        # tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return output.logits, None
        
        
    # def score(
    #     self, y: torch.Tensor, state: Any, x: torch.Tensor
    # ) -> Tuple[torch.Tensor, Any]:
    #     """Score new token.

    #     Args:
    #         y (torch.Tensor): 1D torch.int64 prefix tokens.
    #         state: Scorer state for prefix tokens
    #         x (torch.Tensor): encoder feature that generates ys.

    #     Returns:
    #         tuple[torch.Tensor, Any]: Tuple of
    #             torch.float32 scores for next token (vocab_size)
    #             and next state for ys

    #     """
    #     y = y.unsqueeze(0)
    #     h, _, cache = self.encoder.forward_one_step(
    #         self.embed(y), self._target_mask(y), cache=state
    #     )
    #     h = self.decoder(h[:, -1])
    #     logp = h.log_softmax(dim=-1).squeeze(0)
    #     return logp, cache

    # def batch_score(
    #     self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    # ) -> Tuple[torch.Tensor, List[Any]]:
    #     """Score new token batch.

    #     Args:
    #         ys (torch.Tensor): torch.int64 prefix tokens (n_batch, ylen).
    #         states (List[Any]): Scorer states for prefix tokens.
    #         xs (torch.Tensor):
    #             The encoder feature that generates ys (n_batch, xlen, n_feat).

    #     Returns:
    #         tuple[torch.Tensor, List[Any]]: Tuple of
    #             batchfied scores for next token with shape of `(n_batch, vocab_size)`
    #             and next state list for ys.

    #     """
    #     # merge states
    #     n_batch = len(ys)
    #     n_layers = len(self.encoder.encoders)
    #     if states[0] is None:
    #         batch_state = None
    #     else:
    #         # transpose state of [batch, layer] into [layer, batch]
    #         batch_state = [
    #             torch.stack([states[b][i] for b in range(n_batch)])
    #             for i in range(n_layers)
    #         ]

    #     # batch decoding
    #     h, _, states = self.encoder.forward_one_step(
    #         self.embed(ys), self._target_mask(ys), cache=batch_state
    #     )
    #     h = self.decoder(h[:, -1])
    #     logp = h.log_softmax(dim=-1)

    #     # transpose state of [layer, batch] into [batch, layer]
    #     state_list = [[states[i][b] for i in range(n_layers)] for b in range(n_batch)]
    #     return logp, state_list
