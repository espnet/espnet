import logging
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple

import torch


class DecoderInterface:
    def init_state(self):
        """Initial state for decoding

        Returns:
            dict: initial state
        """
        return dict()

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


def valid_weighted_decoders(
        decoders: Dict[str, DecoderInterface],
        weights: Dict[str, float]
) -> Dict[str, Tuple[DecoderInterface, float]]:
    dec_weights = dict()
    for k, v in decoders.items():
        w = dec_weights.get(k, 1.0)
        if w == 0 or v is None:
            continue
        assert isinstance(v, DecoderInterface), f"{k} ({type(v)}) does not implement DecoderInterface"
        dec_weights[k] = (v, w)
    return dec_weights


class Hypothesis(NamedTuple):
    yseq: List[int] = []
    score: float = 0
    scores: Dict[str, float] = dict()
    states: Dict[str, Dict] = dict()


def beam_search(x, sos, eos, beam_size, decoders, weights,
                token_list=None, maxlenratio=0.0, minlenratio=0.0):
    """Beam search with scorers

    Args:
        x (torch.Tensor): Encoded speech feature (T, D)
        sos (int): Start of sequence id
        eos (int): End of sequence id
        beam_size (int): The number of hypotheses kept during search
        decoders (dict[str, DecoderInterface]): list of decoder modules
        weights (dict[str, float]): List of score weights for each decoders
        token_list (list[str]): List of tokens for debug log
        maxlenratio (float): Input length ratio to obtain max output length.
            If maxlenratio=0.0 (default), it uses a end-detect function
            to automatically find maximum hypothesis lengths
        minlenratio (float): Input length ratio to obtain min output length.

    Returns:
        list: N-best decoding results
    """
    # init decoder states
    dec_weights = valid_weighted_decoders(decoders, weights)
    init_states = dict()
    for k, (d, w) in dec_weights.items():
        init_states[k] = d.init_state()
    init_hyp = dict(score=0.0, scores=dict(), states=init_states, yseq=[sos])

    # set length bounds
    if maxlenratio == 0:
        maxlen = x.shape[0]
    else:
        maxlen = max(1, int(maxlenratio * x.size(0)))
    minlen = int(minlenratio * x.size(0))
    logging.info('max output length: ' + str(maxlen))
    logging.info('min output length: ' + str(minlen))

    # main iteration
    running_hyps = [init_hyp]
    ended_hyps = []
    for i in range(maxlen):
        logging.debug('position ' + str(i))
        hyps_best_kept = []
        for hyp in running_hyps:
            score = 0
            # for k, (d, w) in dec_weights.items():
            #     score += w * d.score(hyp["yseq"], hyp["states"][k], x)
            # top_score, top_ids = score.topk(beam_size)
            # for j in range(beam_size):
            #     new_hyp = dict()


    return ended_hyps
