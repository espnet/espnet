import torch


class DecoderInterface:
    def score(y, state, x):
        """Score new token

        Args:
            y (torch.Tensor): new torch.int64 token to score (B)
            state (dict): decoder state for prefix tokens
            x (torch.Tensor): encoder feature that generates ys (T, D)

        Returns:
            tuple[torch.Tensor, list[dict]]: Tuple of
                torch.float32 scores for y (B)
                and next state for y (B)
        """
        raise NotImplementedError


class LengthBonus(DecoderInterface):
    """Length bonus in beam search"""

    def init_state(self):
        return dict(length=torch.zeros(1))

    def score(y, state, x):
        length = state["length"] + 1
        new_state = dict(length=length)
        n = y.size(0)
        return length.expand(n), [new_state for _ in range(n)]


def beam_search(x, beam_size, decoders, weights, token_list=None):
    """Beam search with scorers

    Args:
        x (torch.Tensor): encoded speech feature (T, D)
        beam_size (int): the number of hypotheses kept during search
        decoders (dict[str, DecoderInterface]): list of decoder modules
        weights (dict[str, float]): list of score weights for each decoders
        token_list (list[str]): list of tokens

    Returns:
        list: N-best decoding results
    """
    return
