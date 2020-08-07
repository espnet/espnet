"""Search algorithms for transducer models."""

import numpy as np

import torch
import torch.nn.functional as F

from espnet.nets.pytorch_backend.transducer.utils import create_lm_batch_state
from espnet.nets.pytorch_backend.transducer.utils import is_prefix
from espnet.nets.pytorch_backend.transducer.utils import recombine_hyps
from espnet.nets.pytorch_backend.transducer.utils import select_lm_state
from espnet.nets.pytorch_backend.transducer.utils import substract


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

    hyp = {
        "score": 0.0,
        "yseq": [decoder.blank],
        "dec_state": dec_state,
        "att_state": None,
    }

    cache = {}

    y, state, _ = decoder.score(hyp, cache, init_tensor)

    for i, hi in enumerate(h):
        ytu = torch.log_softmax(decoder.joint(hi, y[0]), dim=0)
        logp, pred = torch.max(ytu, dim=0)

        if pred != decoder.blank:
            hyp["yseq"].append(int(pred))
            hyp["score"] += float(logp)

            hyp["dec_state"] = state[0]
            hyp["att_state"] = state[1]

            y, state, _ = decoder.score(hyp, cache, init_tensor)

    return [hyp]


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
    beam = recog_args.beam_size
    k_range = min(beam, decoder.odim)

    nbest = recog_args.nbest
    normscore = recog_args.score_norm_transducer

    init_tensor = h.unsqueeze(0)
    dec_state = decoder.init_state(init_tensor)

    kept_hyps = [
        {
            "score": 0.0,
            "yseq": [decoder.blank],
            "dec_state": dec_state,
            "att_state": None,
            "lm_state": None,
        }
    ]

    cache = {}

    for hi in h:
        hyps = kept_hyps
        kept_hyps = []

        while True:
            new_hyp = max(hyps, key=lambda x: x["score"])
            hyps.remove(new_hyp)

            y, state, lm_tokens = decoder.score(new_hyp, cache, init_tensor)

            ytu = F.log_softmax(decoder.joint(hi, y[0]), dim=0)

            if rnnlm:
                rnnlm_state, rnnlm_scores = rnnlm.predict(
                    new_hyp["lm_state"], lm_tokens
                )

            for k in range(decoder.odim):
                beam_hyp = {
                    "score": new_hyp["score"] + float(ytu[k]),
                    "yseq": new_hyp["yseq"][:],
                    "dec_state": new_hyp["dec_state"],
                    "att_state": new_hyp["att_state"],
                    "lm_state": new_hyp["lm_state"],
                }

                if k == decoder.blank:
                    kept_hyps.append(beam_hyp)
                else:
                    beam_hyp["dec_state"] = state[0]
                    beam_hyp["att_state"] = state[1]

                    beam_hyp["yseq"].append(int(k))

                    if rnnlm:
                        beam_hyp["lm_state"] = rnnlm_state
                        beam_hyp["score"] += recog_args.lm_weight * rnnlm_scores[0][k]

                    hyps.append(beam_hyp)

            hyps_max = float(max(hyps, key=lambda x: x["score"])["score"])
            kept_most_prob = len(
                sorted(kept_hyps, key=lambda x: float(x["score"]) > hyps_max)
            )
            if kept_most_prob >= k_range:
                break

    if normscore:
        nbest_hyps = sorted(
            kept_hyps, key=lambda x: x["score"] / len(x["yseq"]), reverse=True
        )[:nbest]
    else:
        nbest_hyps = sorted(kept_hyps, key=lambda x: x["score"], reverse=True)[:nbest]

    return nbest_hyps


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
    beam = recog_args.beam_size
    w_range = min(beam, decoder.odim)

    max_sym_exp = recog_args.nstep
    nbest = recog_args.nbest

    init_tensor = h.unsqueeze(0)

    w_state = (decoder.init_state(torch.zeros((w_range, decoder.dunits))), None)
    dec_state = decoder.select_state(w_state, 0)

    B = [
        {
            "yseq": [decoder.blank],
            "score": 0.0,
            "dec_state": dec_state[0],
            "att_state": None,
            "lm_state": None,
        }
    ]

    if rnnlm:
        if hasattr(rnnlm.predictor, "wordlm"):
            lm_type = "wordlm"
            lm_layers = len(rnnlm.predictor.wordlm.rnn)
        else:
            lm_type = "lm"
            lm_layers = len(rnnlm.predictor.rnn)

    cache = {}

    for hi in h:
        A = []
        C = B

        for v in range(max_sym_exp):
            D = []

            w_y, w_state, w_lm_tokens = decoder.batch_score(
                C, w_state, cache, init_tensor
            )

            h_enc = hi.unsqueeze(0).expand(w_range, -1)

            w_logprobs = F.log_softmax(decoder.joint(h_enc, w_y), dim=-1).view(-1)
            w_blank = w_logprobs[0 :: decoder.odim]

            seq_A = [h["yseq"] for h in A]

            for i, hyp in enumerate(C):
                if hyp["yseq"] not in seq_A:
                    new_hyp = {
                        "score": hyp["score"] + float(w_blank[i]),
                        "yseq": hyp["yseq"][:],
                        "dec_state": hyp["dec_state"],
                        "att_state": hyp["att_state"],
                        "lm_state": hyp["lm_state"],
                    }

                    A.append(new_hyp)
                else:
                    dict_pos = seq_A.index(hyp["yseq"])

                    A[dict_pos]["score"] = np.logaddexp(
                        A[dict_pos]["score"], (hyp["score"] + float(w_blank[i]))
                    )

            if v < max_sym_exp:
                if rnnlm:
                    w_lm_states = create_lm_batch_state(
                        [c["lm_state"] for c in C], lm_type, lm_layers
                    )

                    w_lm_states, w_lm_scores = rnnlm.buff_predict(
                        w_lm_states, w_lm_tokens, len(C)
                    )

                    w_lm_scores = w_lm_scores.contiguous().view(-1)

                for i, hyp in enumerate(C):
                    pos_k = i * decoder.odim
                    k_i = w_logprobs.narrow(0, pos_k, decoder.odim)

                    if rnnlm:
                        lm_k_i = w_lm_scores.narrow(0, pos_k, decoder.odim)

                    for k in range(1, decoder.odim):
                        curr_score = float(k_i[k])

                        beam_hyp = {
                            "score": hyp["score"] + curr_score,
                            "yseq": hyp["yseq"][:],
                            "dec_state": hyp["dec_state"],
                            "att_state": hyp["att_state"],
                            "lm_state": hyp["lm_state"],
                        }

                        beam_hyp["yseq"].append(int(k))

                        new_state = decoder.select_state(w_state, i)

                        beam_hyp["dec_state"] = new_state[0]
                        beam_hyp["att_state"] = new_state[1]

                        if rnnlm:
                            beam_hyp["score"] += recog_args.lm_weight * lm_k_i[k]

                            beam_hyp["lm_state"] = select_lm_state(
                                w_lm_states, i, lm_type, lm_layers
                            )

                        D.append(beam_hyp)

            C = sorted(D, key=lambda x: x["score"], reverse=True)[:w_range]

        B = sorted(A, key=lambda x: x["score"], reverse=True)[:w_range]

    nbest_hyps = sorted(B, key=lambda x: x["score"], reverse=True)[:nbest]

    return nbest_hyps


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
    beam = recog_args.beam_size
    w_range = min(beam, decoder.odim)

    h_length = int(h.size(0))
    u_max = min(recog_args.u_max, (h_length - 1))

    nbest = recog_args.nbest

    init_tensor = h.unsqueeze(0)

    w_state = (decoder.init_state(torch.zeros((w_range, decoder.dunits))), None)

    dec_state = decoder.select_state(w_state, 0)

    B = [
        {
            "yseq": [decoder.blank],
            "score": 0.0,
            "dec_state": dec_state[0],
            "att_state": None,
            "lm_state": None,
        }
    ]
    final = []

    if rnnlm:
        if hasattr(rnnlm.predictor, "wordlm"):
            lm_type = "wordlm"
            lm_layers = len(rnnlm.predictor.wordlm.rnn)
        else:
            lm_type = "lm"
            lm_layers = len(rnnlm.predictor.rnn)

    cache = {}

    for i in range(h_length + u_max):
        A = []

        B_ = []
        h_states = []
        for hyp in B:
            u = len(hyp["yseq"]) - 1
            t = i - u + 1

            if t > (h_length - 1):
                continue

            B_.append(hyp)
            h_states.append((t, h[t]))

        if B_:
            w_y, w_state, w_lm_tokens = decoder.batch_score(
                B_, w_state, cache, init_tensor
            )

            h_enc = torch.stack([h[1] for h in h_states])

            w_logprobs = F.log_softmax(decoder.joint(h_enc, w_y), dim=-1).view(-1)
            w_blank = w_logprobs[0 :: decoder.odim]

            if rnnlm:
                w_lm_states = create_lm_batch_state(
                    [b["lm_state"] for b in B_], lm_type, lm_layers
                )

                w_lm_states, w_lm_scores = rnnlm.buff_predict(
                    w_lm_states, w_lm_tokens, len(B_)
                )

                w_lm_scores = w_lm_scores.contiguous().view(-1)

            for i, hyp in enumerate(B_):
                new_hyp = {
                    "score": hyp["score"] + float(w_blank[i]),
                    "yseq": hyp["yseq"][:],
                    "dec_state": hyp["dec_state"],
                    "att_state": hyp["att_state"],
                    "lm_state": hyp["lm_state"],
                }

                A.append(new_hyp)

                if h_states[i][0] == (h_length - 1):
                    final.append(new_hyp)

                pos_k = i * decoder.odim
                k_i = w_logprobs.narrow(0, pos_k, decoder.odim)

                if rnnlm:
                    lm_k_i = w_lm_scores.narrow(0, pos_k, decoder.odim)

                for k in range(1, decoder.odim):
                    beam_hyp = {
                        "score": hyp["score"] + float(k_i[k]),
                        "yseq": hyp["yseq"][:],
                        "dec_state": hyp["dec_state"],
                        "att_state": hyp["att_state"],
                        "lm_state": hyp["lm_state"],
                    }

                    new_state = decoder.select_state(w_state, i)

                    beam_hyp["dec_state"] = new_state[0]
                    beam_hyp["att_state"] = new_state[1]

                    beam_hyp["yseq"].append(int(k))

                    A.append(beam_hyp)

                    if rnnlm:
                        beam_hyp["score"] += recog_args.lm_weight * lm_k_i[k]

                        beam_hyp["lm_state"] = select_lm_state(
                            w_lm_states, i, lm_type, lm_layers
                        )

        B = sorted(A, key=lambda x: x["score"], reverse=True)[:w_range]
        B = recombine_hyps(B)

    if final:
        nbest_hyps = sorted(final, key=lambda x: x["score"], reverse=True)[:nbest]
    else:
        nbest_hyps = B[:nbest]

    return nbest_hyps


def nsc_beam_search(decoder, h, recog_args, rnnlm=None):
    """N-step constrained beam search implementation.

    Based and modified from https://arxiv.org/pdf/2002.03577.pdf

    Args:
        decoder (class): decoder class
        h (torch.Tensor): encoder hidden state sequences (Tmax, Henc)
        recog_args (Namespace): argument Namespace containing options
        rnnlm (torch.nn.Module): language module

    Returns:
        nbest_hyps (list of dicts): n-best decoding results

    """
    beam = recog_args.beam_size
    w_range = min(beam, decoder.odim)

    nstep = recog_args.nstep
    prefix_alpha = recog_args.prefix_alpha

    nbest = recog_args.nbest

    cache = {}

    init_tensor = h.unsqueeze(0)

    w_state = (decoder.init_state(torch.zeros((w_range, decoder.dunits))), None)
    state = decoder.select_state(w_state, 0)

    init_tokens = [{"yseq": [decoder.blank], "dec_state": state[0], "att_state": None}]

    w_y, w_state, w_lm_tokens = decoder.batch_score(
        init_tokens, w_state, cache, init_tensor
    )

    state = decoder.select_state(w_state, 0)

    if rnnlm:
        w_lm_states, w_lm_scores = rnnlm.buff_predict(None, w_lm_tokens, w_range)

        if hasattr(rnnlm.predictor, "wordlm"):
            lm_type = "wordlm"
            lm_layers = len(rnnlm.predictor.wordlm.rnn)
        else:
            lm_type = "lm"
            lm_layers = len(rnnlm.predictor.rnn)

        lm_state = select_lm_state(w_lm_states, 0, lm_type, lm_layers)
        lm_scores = w_lm_scores[0]
    else:
        lm_state = None
        lm_scores = None

    kept_hyps = [
        {
            "yseq": [decoder.blank],
            "score": 0.0,
            "y": [w_y[0]],
            "dec_state": state[0],
            "att_state": state[1],
            "lm_state": lm_state,
            "lm_scores": lm_scores,
        }
    ]

    for hi in h:
        hyps = sorted(kept_hyps, key=lambda x: len(x["yseq"]), reverse=True)
        kept_hyps = []

        for j in range(len(hyps) - 1):
            for i in range((j + 1), len(hyps)):
                if (
                    is_prefix(hyps[j]["yseq"], hyps[i]["yseq"])
                    and (len(hyps[j]["yseq"]) - len(hyps[i]["yseq"])) <= prefix_alpha
                ):
                    next_id = len(hyps[i]["yseq"])

                    ytu = F.log_softmax(decoder.joint(hi, hyps[i]["y"][-1]), dim=0)

                    curr_score = float(hyps[i]["score"]) + float(
                        ytu[hyps[j]["yseq"][next_id]]
                    )

                    for k in range(next_id, (len(hyps[j]["yseq"]) - 1)):
                        ytu = F.log_softmax(decoder.joint(hi, hyps[j]["y"][k]), dim=0)

                        curr_score += float(ytu[hyps[j]["yseq"][k + 1]])

                    hyps[j]["score"] = np.logaddexp(float(hyps[j]["score"]), curr_score)

        S = []
        V = []
        for n in range(nstep):
            h_enc = hi.unsqueeze(0)
            w_y = torch.stack([hyp["y"][-1] for hyp in hyps])

            w_logprobs = F.log_softmax(decoder.joint(h_enc, w_y), dim=-1).view(-1)

            if rnnlm:
                w_lm_scores = torch.stack([hyp["lm_scores"] for hyp in hyps])
                w_lm_scores = w_lm_scores.contiguous().view(-1)

            for i, hyp in enumerate(hyps):
                pos_k = i * decoder.odim
                k_i = w_logprobs.narrow(0, pos_k, decoder.odim)

                if rnnlm:
                    lm_k_i = w_lm_scores.narrow(0, pos_k, decoder.odim)

                for k in range(decoder.odim):
                    curr_score = float(k_i[k])

                    w_hyp = {
                        "yseq": hyp["yseq"][:],
                        "score": hyp["score"] + curr_score,
                        "y": hyp["y"][:],
                        "dec_state": hyp["dec_state"],
                        "att_state": hyp["att_state"],
                        "lm_state": hyp["lm_state"],
                        "lm_scores": hyp["lm_scores"],
                    }

                    if k == decoder.blank:
                        S.append(w_hyp)
                    else:
                        w_hyp["yseq"].append(int(k))

                        if rnnlm:
                            w_hyp["score"] += recog_args.lm_weight * lm_k_i[k]

                        V.append(w_hyp)

            V = sorted(V, key=lambda x: x["score"], reverse=True)
            V = substract(V, hyps)[:w_range]

            l_state = [(v["dec_state"], v["att_state"]) for v in V]
            l_tokens = [v["yseq"] for v in V]

            w_state = decoder.create_batch_states(w_state, l_state, l_tokens)
            w_y, w_state, w_lm_tokens = decoder.batch_score(
                V, w_state, cache, init_tensor
            )

            if rnnlm:
                w_lm_states = create_lm_batch_state(
                    [v["lm_state"] for v in V], lm_type, lm_layers
                )
                w_lm_states, w_lm_scores = rnnlm.buff_predict(
                    w_lm_states, w_lm_tokens, w_range
                )

            if n < (nstep - 1):
                for i, v in enumerate(V):
                    v["y"].append(w_y[i])

                    new_state = decoder.select_state(w_state, i)

                    v["dec_state"] = new_state[0]
                    v["att_state"] = new_state[1]

                    if rnnlm:
                        v["lm_state"] = select_lm_state(
                            w_lm_states, i, lm_type, lm_layers
                        )
                        v["lm_scores"] = w_lm_scores[i]

                hyps = V[:]
            else:
                w_logprobs = F.log_softmax(decoder.joint(h_enc, w_y), dim=-1).view(-1)
                blank_score = w_logprobs[0 :: decoder.odim]

                for i, v in enumerate(V):
                    if nstep != 1:
                        v["score"] += float(blank_score[i])

                    v["y"].append(w_y[i])

                    new_state = decoder.select_state(w_state, i)

                    v["dec_state"] = new_state[0]
                    v["att_state"] = new_state[1]

                    if rnnlm:
                        v["lm_state"] = select_lm_state(
                            w_lm_states, i, lm_type, lm_layers
                        )
                        v["lm_scores"] = w_lm_scores[i]

        kept_hyps = sorted((S + V), key=lambda x: x["score"], reverse=True)[:w_range]

    nbest_hyps = sorted(
        kept_hyps, key=lambda x: x["score"] / len(x["yseq"]), reverse=True
    )[:nbest]

    return nbest_hyps


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
