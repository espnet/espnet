class ScorerInterface:
    """Decoder interface for beam search"""

    def init_state(self, x):
        """Initial state for decoding

        Args:
            x (torch.Tensor): torch.float32 feature tensor (T, D)

        Returns: initial state
        """
        return None

    def score(self, y, state, x):
        """Score new token

        Args:
            y (torch.Tensor): torch.int64 prefix token (U)
            state: Decoder state for prefix tokens
            x (torch.Tensor): Encoder feature that generates ys (T, D)

        Returns:
            tuple[torch.Tensor, list[dict]]: Tuple of
                torch.float32 scores for next token (n_vocab)
                and next state for ys
        """
        raise NotImplementedError

    def final_score(self, state):
        """Score eos (optional)

        Args:
            state: decoder state for prefix tokens

        Returns:
            float: final score
        """
        return 0.0


class PartialScorerInterface(ScorerInterface):
    """Partial scorer interface for beam search

    The partial scorer performs scoring when non-partial scorer finished scoring,
    and recieves pre-pruned next tokens to score because it is too heavy to score
    all the tokens.
    """

    def select_state(self, state, i):
        """Select state with relative ids in the main beam search

        Args:
            state: Decoder state for prefix tokens
            i (int): Index to select a state in the main beam search

        Returns:
            state: pruned decoder state
        """
        raise NotImplementedError

    def score_partial(self, y, next_tokens, state, x):
        """Score new token

        Args:
            y (torch.Tensor): torch.int64 prefix token (U)
            next_tokens (torch.Tensor): torch.int64 next token to score (N)
            state: decoder state for prefix tokens
            x (torch.Tensor): Encoder feature that generates ys (T, D)

        Returns:
            tuple[torch.Tensor, list[dict]]: Tuple of
                torch.float32 scores for y (N)
                and next state for ys
        """
        raise NotImplementedError
