import logging
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Tuple

import torch

from espnet.nets.e2e_asr_common import end_detect


class DecoderInterface:
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
            state: decoder state for prefix tokens
            x (torch.Tensor): encoder feature that generates ys (T, D)

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


class PartialDecoderInterface(DecoderInterface):
    """Partial decoder interface for beam search

    This performs scoring when non-partial decoders finished scoring,
    and this recieves pre-pruned next tokens to score by `partial_score`
    """

    def select_state(self, state, ids):
        """Select state with relative ids in the final beam

        Args:
            state: decoder state for prefix tokens
            ids (torch.Tensor): relative ids in the final beam

        Returns:
            state: pruned decoder state
        """
        raise NotImplementedError

    def score(self, y, next_tokens, state, x):
        """Score new token

        Args:
            y (torch.Tensor): torch.int64 prefix token (U)
            next_tokens (torch.Tensor): torch.int64 next token to score (N)
            state: decoder state for prefix tokens
            x (torch.Tensor): encoder feature that generates ys (T, D)

        Returns:
            tuple[torch.Tensor, list[dict]]: Tuple of
                torch.float32 scores for y (N)
                and next state for ys
        """
        raise NotImplementedError


class LengthBonus(DecoderInterface):
    """Length bonus in beam search"""

    def __init__(self, n_vocab):
        self.n = n_vocab

    def score(self, y, state, x):
        """Score new token

        Args:
            y (torch.Tensor): torch.int64 prefix token to score (B)
            state: decoder state for prefix tokens
            x (torch.Tensor): encoder feature that generates ys (T, D)

        Returns:
            tuple[torch.Tensor, list[dict]]: Tuple of
                torch.float32 scores for y (B)
                and next state for ys
        """
        import torch
        return torch.tensor([1.0]).expand(self.n), None


def get_weighted_decoders(
        decoders: Dict[str, DecoderInterface],
        weights: Dict[str, float]
) -> Dict[str, Tuple[DecoderInterface, float]]:
    full_dec_weights = dict()
    part_dec_weights = dict()
    """filter invalid weights and decoders"""
    for k, v in decoders.items():
        w = weights.get(k, 1.0)
        if w == 0 or v is None:
            continue
        assert isinstance(v, DecoderInterface), f"{k} ({type(v)}) does not implement DecoderInterface"
        if isinstance(v, PartialDecoderInterface):
            part_dec_weights[k] = (v, w)
        else:
            full_dec_weights[k] = (v, w)
    return full_dec_weights, part_dec_weights


class Hypothesis(NamedTuple):
    """Hypothesis data type"""

    yseq: List[int] = []
    score: float = 0
    scores: Dict[str, float] = dict()
    states: Dict[str, Dict] = dict()


def append_tensor(xs, x):
    x = torch.tensor([x], dtype=xs.dtype, device=xs.device)
    return torch.cat((xs, x))


def beam_search(x, sos, eos, beam_size, decoders, weights,
                token_list=None, maxlenratio=0.0, minlenratio=0.0,
                pre_beam_ratio=1.5, pre_beam_score="decoder"):
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
    # get weighted decoders
    full_dec_weights, part_dec_weights = get_weighted_decoders(decoders, weights)
    all_dec_weights = dict(**full_dec_weights, **part_dec_weights)
    if pre_beam_score not in full_dec_weights or pre_beam_ratio == 0.0:
        logging.warning(f"pre-beam scorer: {pre_beam_score} is not found")

    # init decoder states
    init_states = dict()
    init_scores = dict()
    for k, (d, w) in all_dec_weights.items():
        init_states[k] = d.init_state(x)
        init_scores[k] = 0.0
    init_hyp = Hypothesis(score=0.0, scores=init_scores, states=init_states,
                          yseq=torch.tensor([sos], device=x.device))

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
        best = []
        for hyp in running_hyps:
            scores = dict()
            states = dict()
            wscores = hyp.score
            # scoring full tokens
            for k, (d, w) in full_dec_weights.items():
                scores[k], states[k] = d.score(hyp.yseq, hyp.states[k], x)
                wscores += w * scores[k]

            # pre-beam search
            pre_beam = int(pre_beam_ratio * beam_size)
            local_best_scores, local_best_ids = torch.topk(scores[pre_beam_score], pre_beam)

            # scoring partial tokens
            for k, (d, w) in part_dec_weights.items():
                scores[k], states[k] = d.score(hyp.yseq, local_best_ids, hyp.states[k], x)
                wscores[local_best_ids] += w * scores[k]

            # mask pruned in the pre-beam search
            tmp = wscores[local_best_ids]
            wscores[:] = -float("inf")
            wscores[local_best_ids] = tmp

            # prune hyps
            top_ids = wscores.topk(beam_size)[1]
            # if do_pre_beam:
            local_ids = wscores[local_best_ids].topk(beam_size)[1]
            for j, local_j in zip(top_ids, local_ids):
                j = int(j)

                new_scores = dict()
                for k in full_dec_weights:
                    new_scores[k] = float(hyp.scores[k] + scores[k][j])
                for k in part_dec_weights:
                    new_scores[k] = scores[k][local_j]

                new_states = dict()
                for k in full_dec_weights:
                    new_states[k] = states[k]
                for k in part_dec_weights:
                    new_states[k] = decoders[k].select_state(states[k], local_j)

                # will be (2 x beam at most)
                best.append(Hypothesis(
                    score=float(wscores[j]),
                    yseq=append_tensor(hyp.yseq, j),
                    scores=new_scores,
                    states=new_states))
            best = sorted(best, key=lambda x: x.score, reverse=True)[:beam_size]

        running_hyps = best
        logging.debug(f'the number of running hypothes: {len(running_hyps)}')
        if token_list is not None:
            logging.debug("best hypo: " + "".join(
                [token_list[x] for x in running_hyps[0].yseq[1:]]))
        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1:
            logging.info("adding <eos> in the last position in the loop")
            running_hyps = [h._replace(yseq=append_tensor(h.yseq, eos)) for h in running_hyps]

        # add ended hypothes to a final list, and removed them from current hypothes
        # (this will be a probmlem, number of hyps < beam)
        remained_hyps = []
        for hyp in running_hyps:
            if hyp.yseq[-1] == eos:
                # e.g., Word LM needs to add final <eos> score
                for k, (d, w) in all_dec_weights.items():
                    s = d.final_score(hyp.states[k])
                    hyp.scores[k] += s
                    hyp = hyp._replace(score=hyp.score + w * s)
                ended_hyps.append(hyp)
            else:
                remained_hyps.append(hyp)
        # end detection
        if maxlenratio == 0.0 and end_detect(ended_hyps, i):
            logging.info(f'end detected at {i}')
            break
        running_hyps = remained_hyps
        if len(running_hyps) > 0:
            logging.debug(f'remeined hypothes: {len(running_hyps)}')
        else:
            logging.info('no hypothesis. Finish decoding.')
            break

    nbest_hyps = sorted(ended_hyps, key=lambda x: x.score, reverse=True)[:beam_size]
    # check number of hypotheis
    if len(nbest_hyps) == 0:
        logging.warning('there is no N-best results, perform recognition again with smaller minlenratio.')
        return beam_search(x=x, sos=sos, eos=eos, beam_size=beam_size,
                           weights=weights, decoders=decoders,
                           token_list=token_list,
                           maxlenratio=maxlenratio,
                           minlenratio=max(0.0, minlenratio - 0.1))

    best = nbest_hyps[0]
    logging.info(f'total log probability: {best.score}')
    logging.info(f'normalized log probability: {best.score / len(best.yseq)}')
    return [h._replace(yseq=h.yseq.tolist())._asdict() for h in nbest_hyps]
