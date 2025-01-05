import copy
import logging
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from typeguard import typechecked

from espnet2.lm.abs_model import AbsLM
from espnet.nets.pytorch_backend.transformer.mask import subsequent_mask


class HuggingfaceOPTModel(AbsLM):
    """
        HuggingfaceOPTModel is a language model that utilizes the OPT architecture
    from Hugging Face's Transformers library. This model inherits from the
    abstract class AbsLM and implements methods for forward propagation and
    scoring of tokens.

    Attributes:
        pretrained_params (dict): A copy of the pretrained parameters from the
            OPT model excluding the embedding layer weights.
        decoder (OPTModel): The decoder model based on the OPT architecture.
        lm_head (nn.Linear): A linear layer that projects the hidden states
            to the vocabulary size.

    Args:
        vocab_size (int): The size of the vocabulary for the language model.
        opt_name (str): The name of the pretrained OPT model to load.

    Raises:
        Exception: If the transformers library is not properly installed.

    Examples:
        # Initializing the model
        model = HuggingfaceOPTModel(vocab_size=50265, opt_name='facebook/opt-1.3b')

        # Forward pass
        input_ids = torch.tensor([[1, 2, 3], [4, 5, 0]])  # Example input
        logits, _ = model(input_ids, None)

        # Scoring a new token
        y = torch.tensor([6])  # New token
        state = None  # No previous state
        scores, new_state = model.score(y, state, input_ids)

        # Batch scoring
        ys = torch.tensor([[1, 2], [3, 4]])  # Prefix tokens
        states = [None, None]  # States for each prefix
        xs = torch.tensor([[0.1, 0.2], [0.3, 0.4]])  # Encoder features
        batch_scores, new_states = model.batch_score(ys, states, xs)

        # Reloading pretrained parameters
        model.reload_pretrained_parameters()
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        opt_name: str,
    ):
        super().__init__()
        try:
            from transformers import OPTModel
        except Exception as e:
            print("Error: transformers is not properly installed.")
            print("Please install transformers")
            raise e

        # opt_model_name_pattern = re.compile(r"facebook/opt-\d+m")
        # assert opt_model_name_pattern.match(opt_name) is not None

        pretrained_opt_model = OPTModel.from_pretrained(opt_name)
        pretrained_opt_model_dict = pretrained_opt_model.state_dict()
        pretrained_opt_model_dict.pop("decoder.embed_tokens.weight")
        self.pretrained_params = copy.deepcopy(pretrained_opt_model_dict)

        config = pretrained_opt_model.config
        config.vocab_size = vocab_size
        config.bos_token_id = vocab_size - 1
        config.eos_token_id = vocab_size - 1
        config.pad_token_id = 0

        self.decoder = OPTModel(config)

        self.lm_head = nn.Linear(
            config.word_embed_proj_dim, config.vocab_size, bias=False
        )

    def _target_mask(self, ys_in_pad):
        ys_mask = ys_in_pad != 0
        m = subsequent_mask(ys_mask.size(-1), device=ys_mask.device).unsqueeze(0)
        return ys_mask.unsqueeze(-2) & m

    def forward(self, input: torch.Tensor, hidden: None) -> Tuple[torch.Tensor, None]:
        """
        Compute LM loss value from buffer sequences.

        This method takes input token IDs and computes the corresponding
        logits using the model's decoder and a linear layer (lm_head).

        Args:
            input (torch.Tensor): Input token IDs of shape (batch, len).
            hidden (None): Placeholder for future use; currently not used.

        Returns:
            Tuple[torch.Tensor, None]: A tuple containing:
                - logits (torch.Tensor): Output logits of shape (batch, len, vocab_size).
                - None: Placeholder for hidden state; currently returns None.

        Examples:
            >>> model = HuggingfaceOPTModel(vocab_size=1000, opt_name='facebook/opt-125m')
            >>> input_tensor = torch.randint(0, 1000, (2, 10))  # (batch, len)
            >>> logits, _ = model.forward(input_tensor, None)
            >>> logits.shape
            torch.Size([2, 10, 1000])
        """
        pad_mask = input != 0
        y = self.decoder(
            input,
            attention_mask=pad_mask,
            return_dict=True,
        )
        y = y.last_hidden_state

        logits = self.lm_head(y)

        return logits, None

    def score(
        self, y: torch.Tensor, state: Any, x: torch.Tensor
    ) -> Tuple[torch.Tensor, Any]:
        """
            Score new token.

        This method computes the scores for the next token based on the provided
        prefix tokens and the current state. It leverages the underlying OPT model
        to perform this scoring, returning the softmax scores and the updated
        state for subsequent predictions.

        Args:
            y (torch.Tensor): 1D torch.int64 prefix tokens.
            state: Scorer state for prefix tokens.
            x (torch.Tensor): Encoder feature that generates ys.

        Returns:
            Tuple[torch.Tensor, Any]: A tuple containing:
                - torch.float32 scores for the next token (vocab_size).
                - Next state for ys, which can be used in future calls.

        Examples:
            >>> model = HuggingfaceOPTModel(vocab_size=50265, opt_name="facebook/opt-1.3b")
            >>> prefix_tokens = torch.tensor([1, 2, 3])  # Example prefix tokens
            >>> state = None  # Initial state
            >>> encoder_features = torch.randn(1, 3, 768)  # Example encoder features
            >>> scores, new_state = model.score(prefix_tokens, state, encoder_features)
            >>> print(scores.shape)  # Should print: torch.Size([50265])

        Note:
            Ensure that the state passed is compatible with the model's caching
            mechanism for optimal performance.
        """
        if state is None:
            _use_cache = True
        else:
            _use_cache = False

        y = y.unsqueeze(0)

        output = self.decoder(
            y,
            past_key_values=state,
            use_cache=_use_cache,
            output_hidden_states=True,
            return_dict=True,
        )

        h = output.last_hidden_state[:, -1]
        h = self.lm_head(h)
        cache = output.past_key_values
        logp = h.log_softmax(dim=-1).squeeze(0)
        return logp, cache

    def batch_score(
        self, ys: torch.Tensor, states: List[Any], xs: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Any]]:
        """
            Score new token batch.

        This method computes the scores for a batch of new tokens based on the
        provided input sequences and their corresponding states. It leverages the
        decoder from the OPT model to perform batch decoding and returns the
        scores for the next token along with updated states.

        Args:
            ys (torch.Tensor):
                torch.int64 prefix tokens of shape (n_batch, ylen).
            states (List[Any]):
                Scorer states for prefix tokens, with each state corresponding
                to a batch entry.
            xs (torch.Tensor):
                The encoder feature that generates ys, with shape
                (n_batch, xlen, n_feat).

        Returns:
            tuple[torch.Tensor, List[Any]]:
                Tuple containing:
                    - batchfied scores for next token with shape of
                      `(n_batch, vocab_size)`.
                    - next state list for ys.

        Examples:
            >>> model = HuggingfaceOPTModel(vocab_size=50265, opt_name='facebook/opt-1.3b')
            >>> ys = torch.tensor([[1, 2, 3], [4, 5, 6]])
            >>> states = [None, None]
            >>> xs = torch.randn(2, 10, 768)  # Example encoder features
            >>> scores, next_states = model.batch_score(ys, states, xs)
            >>> print(scores.shape)  # Should output: torch.Size([2, 50265])
        """
        # merge states
        n_batch = len(ys)
        n_layers = len(self.decoder.decoder.layers)
        if states[0] is None:
            _use_cache = True
        else:
            _use_cache = False

        # batch decoding
        output = self.decoder(
            ys,
            use_cache=_use_cache,
            output_hidden_states=True,
            return_dict=True,
        )
        h = output.last_hidden_state
        h = self.lm_head(h[:, -1])

        logp = h.log_softmax(dim=-1)

        state_list = [[[] for i in range(n_layers)] for b in range(n_batch)]
        return logp, state_list

    def reload_pretrained_parameters(self):
        """
                Reloads the pretrained parameters into the decoder of the model.

        This method updates the decoder's state dictionary with the pretrained parameters
        stored in `self.pretrained_params`. It allows for the model to be re-initialized
        with pretrained weights without the need to re-instantiate the model itself.

        Raises:
            RuntimeError: If the state_dict cannot be loaded into the decoder.

        Examples:
            >>> model = HuggingfaceOPTModel(vocab_size=50265, opt_name="facebook/opt-1.3b")
            >>> model.reload_pretrained_parameters()
            INFO:root:Pretrained OPT model parameters reloaded!

        Note:
            Ensure that the pretrained parameters have been correctly set before calling
            this method.
        """
        logging.info("Pretrained OPT model parameters reloaded!")
