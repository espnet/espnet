"""Search algorithms for transducer models."""

import numpy as np

from dataclasses import asdict
from dataclasses import dataclass

from typing import Any
from typing import Dict
from typing import List
from typing import Union

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.transducer.utils import create_lm_batch_state
from espnet.nets.pytorch_backend.transducer.utils import init_lm_state
from espnet.nets.pytorch_backend.transducer.utils import is_prefix
from espnet.nets.pytorch_backend.transducer.utils import recombine_hyps
from espnet.nets.pytorch_backend.transducer.utils import select_lm_state
from espnet.nets.pytorch_backend.transducer.utils import substract


@dataclass
class Hypothesis:
    """Hypothesis class for beam search algorithms."""

    score: float
    yseq: List[int]
    dec_state: Union[List[List[torch.Tensor]], List[torch.Tensor]]
    y: List[torch.tensor] = None
    lm_state: Union[Dict[str, Any], List[Any]] = None
    lm_scores: torch.Tensor = None


def greedy_search(decoder, h, recog_args):
    """Greedy search implementation for transformer-transducer.

    Args:
        decoder (class): decoder class
        h (torch.Tensor): encoder hidden state sequences (maxlen_in, Henc)
        recog_args (Namespace): argument Namespace containing options

    Returns:
        hyp (list of dicts): 1-best decoding results

    """
    init_tensor = h.unsqueeze(0)
    dec_state = decoder.init_state(init_tensor)

    hyp = Hypothesis(score=0.0, yseq=[decoder.blank], dec_state=dec_state)

    cache = {}

    y, state, _ = decoder.score(hyp, cache, init_tensor)

    for i, hi in enumerate(h):
        ytu = torch.log_softmax(decoder.joint(hi, y[0]), dim=-1)
        logp, pred = torch.max(ytu, dim=-1)

        if pred != decoder.blank:
            hyp.yseq.append(int(pred))
            hyp.score += float(logp)

            hyp.dec_state = state

            y, state, _ = decoder.score(hyp, cache, init_tensor)

    return [asdict(hyp)]


def default_beam_search(decoder, h, recog_args, rnnlm=None):
    """Beam search implementation.

    Args:
        decoder (class): decoder class
        h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
        recog_args (Namespace): argument Namespace containing options
        rnnlm (torch.nn.Module): language module

    Returns:
        nbest_hyps (list of dicts): n-best decoding results

    """
    beam = min(recog_args.beam_size, decoder.odim)
    beam_k = min(beam, (decoder.odim - 1))

    nbest = recog_args.nbest
    normscore = recog_args.score_norm_transducer

    init_tensor = h.unsqueeze(0)
    blank_tensor = init_tensor.new_zeros(1, dtype=torch.long)

    dec_state = decoder.init_state(init_tensor)

    kept_hyps = [Hypothesis(score=0.0, yseq=[decoder.blank], dec_state=dec_state)]

    cache = {}

    for hi in h:
        hyps = kept_hyps
        kept_hyps = []

        while True:
            max_hyp = max(hyps, key=lambda x: x.score)
            hyps.remove(max_hyp)

            y, state, lm_tokens = decoder.score(max_hyp, cache, init_tensor)

            ytu = F.log_softmax(decoder.joint(hi, y[0]), dim=-1)

            top_k = ytu[1:].topk(beam_k, dim=-1)

            ytu = (
                torch.cat((top_k[0], ytu[0:1])),
                torch.cat((top_k[1] + 1, blank_tensor)),
            )

            if rnnlm:
                rnnlm_state, rnnlm_scores = rnnlm.predict(max_hyp.lm_state, lm_tokens)

            for logp, k in zip(*ytu):
                new_hyp = Hypothesis(
                    score=(max_hyp.score + float(logp)),
                    yseq=max_hyp.yseq[:],
                    dec_state=max_hyp.dec_state,
                    lm_state=max_hyp.lm_state,
                )

                if k == decoder.blank:
                    kept_hyps.append(new_hyp)
                else:
                    new_hyp.dec_state = state

                    new_hyp.yseq.append(int(k))

                    if rnnlm:
                        new_hyp.lm_state = rnnlm_state
                        new_hyp.score += recog_args.lm_weight * rnnlm_scores[0][k]

                    hyps.append(new_hyp)

            hyps_max = float(max(hyps, key=lambda x: x.score).score)
            kept_most_prob = sorted(
                [hyp for hyp in kept_hyps if hyp.score > hyps_max],
                key=lambda x: x.score,
            )
            if len(kept_most_prob) >= beam:
                kept_hyps = kept_most_prob
                break

    if normscore:
        nbest_hyps = sorted(
            kept_hyps, key=lambda x: x.score / len(x.yseq), reverse=True
        )[:nbest]
    else:
        nbest_hyps = sorted(kept_hyps, key=lambda x: x.score, reverse=True)[:nbest]

    return [asdict(n) for n in nbest_hyps]


def time_sync_decoding(decoder, h, recog_args, rnnlm=None):
    """Time synchronous beam search implementation.

    Based on https://ieeexplore.ieee.org/document/9053040

    Args:
        decoder (class): decoder class
        h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
        recog_args (Namespace): argument Namespace containing options
        rnnlm (torch.nn.Module): language module

    Returns:
        nbest_hyps (list of dicts): n-best decoding results

    """
    beam = min(recog_args.beam_size, decoder.odim)

    max_sym_exp = recog_args.max_sym_exp
    nbest = recog_args.nbest

    init_tensor = h.unsqueeze(0)

    beam_state = decoder.init_state(torch.zeros((beam, decoder.dunits)))

    B = [
        Hypothesis(
            yseq=[decoder.blank],
            score=0.0,
            dec_state=decoder.select_state(beam_state, 0),
        )
    ]

    if rnnlm:
        if hasattr(rnnlm.predictor, "wordlm"):
            lm_model = rnnlm.predictor.wordlm
            lm_type = "wordlm"
        else:
            lm_model = rnnlm.predictor
            lm_type = "lm"

            B[0].lm_state = init_lm_state(lm_model)

        lm_layers = len(lm_model.rnn)

    cache = {}

    for hi in h:
        A = []
        C = B

        h_enc = hi.unsqueeze(0)

        for v in range(max_sym_exp):
            D = []

            beam_y, beam_state, beam_lm_tokens = decoder.batch_score(
                C, beam_state, cache, init_tensor
            )

            beam_logp = F.log_softmax(decoder.joint(h_enc, beam_y), dim=-1)
            beam_topk = beam_logp[:, 1:].topk(beam, dim=-1)

            seq_A = [h.yseq for h in A]

            for i, hyp in enumerate(C):
                if hyp.yseq not in seq_A:
                    A.append(
                        Hypothesis(
                            score=(hyp.score + float(beam_logp[i, 0])),
                            yseq=hyp.yseq[:],
                            dec_state=hyp.dec_state,
                            lm_state=hyp.lm_state,
                        )
                    )
                else:
                    dict_pos = seq_A.index(hyp.yseq)

                    A[dict_pos].score = np.logaddexp(
                        A[dict_pos].score, (hyp.score + float(beam_logp[i, 0]))
                    )

            if v < max_sym_exp:
                if rnnlm:
                    beam_lm_states = create_lm_batch_state(
                        [c.lm_state for c in C], lm_type, lm_layers
                    )

                    beam_lm_states, beam_lm_scores = rnnlm.buff_predict(
                        beam_lm_states, beam_lm_tokens, len(C)
                    )

                for i, hyp in enumerate(C):
                    for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                        new_hyp = Hypothesis(
                            score=(hyp.score + float(logp)),
                            yseq=(hyp.yseq + [int(k)]),
                            dec_state=decoder.select_state(beam_state, i),
                            lm_state=hyp.lm_state,
                        )

                        if rnnlm:
                            new_hyp.score += recog_args.lm_weight * beam_lm_scores[i, k]

                            new_hyp.lm_state = select_lm_state(
                                beam_lm_states, i, lm_type, lm_layers
                            )

                        D.append(new_hyp)

            C = sorted(D, key=lambda x: x.score, reverse=True)[:beam]

        B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]

    nbest_hyps = sorted(B, key=lambda x: x.score, reverse=True)[:nbest]

    return [asdict(n) for n in nbest_hyps]


def align_length_sync_decoding(decoder, h, recog_args, rnnlm=None):
    """Alignment-length synchronous beam search implementation.

    Based on https://ieeexplore.ieee.org/document/9053040

    Args:
        decoder (class): decoder class
        h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
        recog_args (Namespace): argument Namespace containing options
        rnnlm (torch.nn.Module): language module

    Returns:
        nbest_hyps (list of dicts): n-best decoding results

    """
    beam = min(recog_args.beam_size, decoder.odim)

    h_length = int(h.size(0))
    u_max = min(recog_args.u_max, (h_length - 1))

    nbest = recog_args.nbest

    init_tensor = h.unsqueeze(0)

    beam_state = decoder.init_state(torch.zeros((beam, decoder.dunits)))

    B = [
        Hypothesis(
            yseq=[decoder.blank],
            score=0.0,
            dec_state=decoder.select_state(beam_state, 0),
        )
    ]
    final = []

    if rnnlm:
        if hasattr(rnnlm.predictor, "wordlm"):
            lm_model = rnnlm.predictor.wordlm
            lm_type = "wordlm"
        else:
            lm_model = rnnlm.predictor
            lm_type = "lm"

            B[0].lm_state = init_lm_state(lm_model)

        lm_layers = len(lm_model.rnn)

    cache = {}

    for i in range(h_length + u_max):
        A = []

        B_ = []
        h_states = []
        for hyp in B:
            u = len(hyp.yseq) - 1
            t = i - u + 1

            if t > (h_length - 1):
                continue

            B_.append(hyp)
            h_states.append((t, h[t]))

        if B_:
            beam_y, beam_state, beam_lm_tokens = decoder.batch_score(
                B_, beam_state, cache, init_tensor
            )

            h_enc = torch.stack([h[1] for h in h_states])

            beam_logp = F.log_softmax(decoder.joint(h_enc, beam_y), dim=-1)
            beam_topk = beam_logp[:, 1:].topk(beam, dim=-1)

            if rnnlm:
                beam_lm_states = create_lm_batch_state(
                    [b.lm_state for b in B_], lm_type, lm_layers
                )

                beam_lm_states, beam_lm_scores = rnnlm.buff_predict(
                    beam_lm_states, beam_lm_tokens, len(B_)
                )

            for i, hyp in enumerate(B_):
                new_hyp = Hypothesis(
                    score=(hyp.score + float(beam_logp[i, 0])),
                    yseq=hyp.yseq[:],
                    dec_state=hyp.dec_state,
                    lm_state=hyp.lm_state,
                )

                A.append(new_hyp)

                if h_states[i][0] == (h_length - 1):
                    final.append(new_hyp)

                for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                    new_hyp = Hypothesis(
                        score=(hyp.score + float(logp)),
                        yseq=(hyp.yseq[:] + [int(k)]),
                        dec_state=decoder.select_state(beam_state, i),
                        lm_state=hyp.lm_state,
                    )

                    if rnnlm:
                        new_hyp.score += recog_args.lm_weight * beam_lm_scores[i, k]

                        new_hyp.lm_state = select_lm_state(
                            beam_lm_states, i, lm_type, lm_layers
                        )

                    A.append(new_hyp)

            B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]
            B = recombine_hyps(B)

    if final:
        nbest_hyps = sorted(final, key=lambda x: x.score, reverse=True)[:nbest]
    else:
        nbest_hyps = B[:nbest]

    return [asdict(n) for n in nbest_hyps]


def nsc_beam_search(decoder, h, recog_args, rnnlm=None):
    """N-step constrained beam search implementation.

    Based and modified from https://arxiv.org/pdf/2002.03577.pdf.
    Please reference ESPnet (b-flo, PR #2444) for any usage outside ESPnet
    until further modifications.

    Note: the algorithm is not in his "complete" form but works almost as
          intended.

    Args:
        decoder (class): decoder class
        h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
        recog_args (Namespace): argument Namespace containing options
        rnnlm (torch.nn.Module): language module

    Returns:
        nbest_hyps (list of dicts): n-best decoding results

    """
    beam = min(recog_args.beam_size, decoder.odim)
    beam_k = min(beam, (decoder.odim - 1))

    nstep = recog_args.nstep
    prefix_alpha = recog_args.prefix_alpha

    nbest = recog_args.nbest

    cache = {}

    init_tensor = h.unsqueeze(0)
    blank_tensor = init_tensor.new_zeros(1, dtype=torch.long)

    beam_state = decoder.init_state(torch.zeros((beam, decoder.dunits)))

    init_tokens = [
        Hypothesis(
            yseq=[decoder.blank],
            score=0.0,
            dec_state=decoder.select_state(beam_state, 0),
        )
    ]

    beam_y, beam_state, beam_lm_tokens = decoder.batch_score(
        init_tokens, beam_state, cache, init_tensor
    )

    state = decoder.select_state(beam_state, 0)

    if rnnlm:
        beam_lm_states, beam_lm_scores = rnnlm.buff_predict(None, beam_lm_tokens, 1)

        if hasattr(rnnlm.predictor, "wordlm"):
            lm_model = rnnlm.predictor.wordlm
            lm_type = "wordlm"
        else:
            lm_model = rnnlm.predictor
            lm_type = "lm"

        lm_layers = len(lm_model.rnn)

        lm_state = select_lm_state(beam_lm_states, 0, lm_type, lm_layers)
        lm_scores = beam_lm_scores[0]
    else:
        lm_state = None
        lm_scores = None

    kept_hyps = [
        Hypothesis(
            yseq=[decoder.blank],
            score=0.0,
            dec_state=state,
            y=[beam_y[0]],
            lm_state=lm_state,
            lm_scores=lm_scores,
        )
    ]

    for hi in h:
        hyps = sorted(kept_hyps, key=lambda x: len(x.yseq), reverse=True)
        kept_hyps = []

        h_enc = hi.unsqueeze(0)

        for j in range(len(hyps) - 1):
            for i in range((j + 1), len(hyps)):
                if (
                    is_prefix(hyps[j].yseq, hyps[i].yseq)
                    and (len(hyps[j].yseq) - len(hyps[i].yseq)) <= prefix_alpha
                ):
                    next_id = len(hyps[i].yseq)

                    ytu = F.log_softmax(decoder.joint(hi, hyps[i].y[-1]), dim=0)

                    curr_score = hyps[i].score + float(ytu[hyps[j].yseq[next_id]])

                    for k in range(next_id, (len(hyps[j].yseq) - 1)):
                        ytu = F.log_softmax(decoder.joint(hi, hyps[j].y[k]), dim=0)

                        curr_score += float(ytu[hyps[j].yseq[k + 1]])

                    hyps[j].score = np.logaddexp(hyps[j].score, curr_score)

        S = []
        V = []
        for n in range(nstep):
            beam_y = torch.stack([hyp.y[-1] for hyp in hyps])

            beam_logp = F.log_softmax(decoder.joint(h_enc, beam_y), dim=-1)
            beam_topk = beam_logp[:, 1:].topk(beam_k, dim=-1)

            if rnnlm:
                beam_lm_scores = torch.stack([hyp.lm_scores for hyp in hyps])

            for i, hyp in enumerate(hyps):
                i_topk = (
                    torch.cat((beam_topk[0][i], beam_logp[i, 0:1])),
                    torch.cat((beam_topk[1][i] + 1, blank_tensor)),
                )

                for logp, k in zip(*i_topk):
                    new_hyp = Hypothesis(
                        yseq=hyp.yseq[:],
                        score=(hyp.score + float(logp)),
                        y=hyp.y[:],
                        dec_state=hyp.dec_state,
                        lm_state=hyp.lm_state,
                        lm_scores=hyp.lm_scores,
                    )

                    if k == decoder.blank:
                        S.append(new_hyp)
                    else:
                        new_hyp.yseq.append(int(k))

                        if rnnlm:
                            new_hyp.score += recog_args.lm_weight * float(
                                beam_lm_scores[i, k]
                            )

                        V.append(new_hyp)

            V = sorted(V, key=lambda x: x.score, reverse=True)
            V = substract(V, hyps)[:beam]

            l_state = [v.dec_state for v in V]
            l_tokens = [v.yseq for v in V]

            beam_state = decoder.create_batch_states(beam_state, l_state, l_tokens)
            beam_y, beam_state, beam_lm_tokens = decoder.batch_score(
                V, beam_state, cache, init_tensor
            )

            if rnnlm:
                beam_lm_states = create_lm_batch_state(
                    [v.lm_state for v in V], lm_type, lm_layers
                )
                beam_lm_states, beam_lm_scores = rnnlm.buff_predict(
                    beam_lm_states, beam_lm_tokens, len(V)
                )

            if n < (nstep - 1):
                for i, v in enumerate(V):
                    v.y.append(beam_y[i])

                    v.dec_state = decoder.select_state(beam_state, i)

                    if rnnlm:
                        v.lm_state = select_lm_state(
                            beam_lm_states, i, lm_type, lm_layers
                        )
                        v.lm_scores = beam_lm_scores[i]

                hyps = V[:]
            else:
                beam_logp = F.log_softmax(decoder.joint(h_enc, beam_y), dim=-1)

                for i, v in enumerate(V):
                    if nstep != 1:
                        v.score += float(beam_logp[i, 0])

                    v.y.append(beam_y[i])

                    v.dec_state = decoder.select_state(beam_state, i)

                    if rnnlm:
                        v.lm_state = select_lm_state(
                            beam_lm_states, i, lm_type, lm_layers
                        )
                        v.lm_scores = beam_lm_scores[i]

        kept_hyps = sorted((S + V), key=lambda x: x.score, reverse=True)[:beam]

    nbest_hyps = sorted(kept_hyps, key=lambda x: (x.score / len(x.yseq)), reverse=True)[
        :nbest
    ]

    return [asdict(n) for n in nbest_hyps]


def search_interface(decoder, h, recog_args, rnnlm):
    """Select and run search algorithms.

    Args:
        decoder (class): decoder class
        h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
        recog_args (Namespace): argument Namespace containing options
        rnnlm (torch.nn.Module): language module

    Returns:
        nbest_hyps (list of dicts): n-best decoding results

    """
    if hasattr(decoder, "att"):
        decoder.att[0].reset()

    if recog_args.beam_size <= 1:
        nbest_hyps = greedy_search(decoder, h, recog_args)
    elif recog_args.search_type == "default":
        nbest_hyps = default_beam_search(decoder, h, recog_args, rnnlm)
    elif recog_args.search_type == "nsc":
        nbest_hyps = nsc_beam_search(decoder, h, recog_args, rnnlm)
    elif recog_args.search_type == "tsd":
        nbest_hyps = time_sync_decoding(decoder, h, recog_args, rnnlm)
    elif recog_args.search_type == "alsd":
        nbest_hyps = align_length_sync_decoding(decoder, h, recog_args, rnnlm)
    else:
        raise NotImplementedError

    return nbest_hyps
