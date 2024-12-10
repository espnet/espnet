"""Search algorithms for Transducer models."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr_transducer.joint_network import JointNetwork
from espnet2.lm.transformer_lm import TransformerLM
from espnet.nets.beam_search import Hypothesis as Hypothesis2
from espnet.nets.pytorch_backend.transducer.utils import (
    is_prefix,
    recombine_hyps,
    select_k_expansions,
    subtract,
)


@dataclass
class Hypothesis:
    """
    Default hypothesis definition for Transducer search algorithms.

    This class represents a single hypothesis in the context of Transducer
    models during beam search decoding. It includes the score of the
    hypothesis, the sequence of tokens generated so far, and the state
    information for both the decoder and language model.

    Attributes:
        score (float): The score of the hypothesis, which is a cumulative
            measure of its likelihood.
        yseq (List[int]): A list of integers representing the generated
            token sequence.
        dec_state (Union[Tuple[torch.Tensor, Optional[torch.Tensor]],
            List[Optional[torch.Tensor]], torch.Tensor]): The state of the
            decoder, which may vary depending on the decoder's architecture.
        lm_state (Union[Dict[str, Any], List[Any]]): The state of the
            language model, if applicable. Defaults to None.

    Examples:
        >>> hyp = Hypothesis(score=0.5, yseq=[1, 2, 3], dec_state=torch.zeros(1, 256))
        >>> print(hyp.score)
        0.5
        >>> print(hyp.yseq)
        [1, 2, 3]
    """

    score: float
    yseq: List[int]
    dec_state: Union[
        Tuple[torch.Tensor, Optional[torch.Tensor]],
        List[Optional[torch.Tensor]],
        torch.Tensor,
    ]
    lm_state: Union[Dict[str, Any], List[Any]] = None


@dataclass
class ExtendedHypothesis(Hypothesis):
    """
    Extended hypothesis definition for NSC beam search and mAES.

    This class extends the base `Hypothesis` class to include additional
    attributes necessary for more sophisticated beam search algorithms
    such as N-step constrained (NSC) beam search and modified adaptive
    expansion search (mAES). It provides storage for the decoder output
    and language model scores, which are essential for evaluating and
    selecting hypotheses during the search process.

    Attributes:
        dec_out (List[torch.Tensor]): A list of tensors representing the
            decoder outputs for the current hypothesis.
        lm_scores (torch.Tensor): A tensor holding the scores from the
            language model for the current hypothesis.

    Examples:
        >>> hypothesis = ExtendedHypothesis(
        ...     score=1.0,
        ...     yseq=[2, 3, 4],
        ...     dec_state=(torch.tensor([0.1]), None),
        ...     dec_out=[torch.tensor([0.2]), torch.tensor([0.3])],
        ...     lm_scores=torch.tensor([0.5, 0.6])
        ... )
        >>> print(hypothesis.yseq)
        [2, 3, 4]
        >>> print(hypothesis.score)
        1.0
        >>> print(hypothesis.dec_out)
        [tensor(0.2), tensor(0.3)]
    """

    dec_out: List[torch.Tensor] = None
    lm_scores: torch.Tensor = None


class BeamSearchTransducerStreaming:
    """
    Beam search implementation for Transducer models.

    This class performs beam search decoding for Transducer models,
    leveraging various search strategies including greedy, time-synchronous,
    and constrained beam search methods. It integrates an optional language
    model for enhanced decoding performance and allows customization of
    multiple parameters to tailor the search behavior.

    Attributes:
        decoder: An instance of AbsDecoder used for generating predictions.
        joint_network: An instance of JointNetwork used for joint decoding.
        beam_size: The number of hypotheses to maintain during search.
        hidden_size: The size of the hidden states in the decoder.
        vocab_size: The size of the vocabulary.
        sos: The start-of-sequence token ID.
        token_list: An optional list of tokens for decoding output.
        blank_id: The ID of the blank token used in Transducer models.
        penalty: The penalty applied during decoding to adjust scores.
        search_algorithm: The selected search algorithm for decoding.
        use_lm: A boolean indicating if a language model is used.
        lm: The language model used for scoring hypotheses.
        lm_weight: Weighting factor for the language model's contribution.
        score_norm: A boolean indicating whether to normalize scores.
        score_norm_during: A boolean indicating if scores should be normalized during search.
        nbest: The number of best hypotheses to return.
        hold_n: The number of tokens to hold for incremental decoding.

    Args:
        decoder: Decoder module.
        joint_network: Joint network module.
        beam_size: Beam size.
        lm: Language model class (optional).
        lm_weight: Weight for soft fusion with the language model (default: 0.1).
        search_type: Type of search algorithm to use during inference.
        max_sym_exp: Maximum number of symbol expansions at each time step (default: 2).
        u_max: Maximum output sequence length (default: 50).
        nstep: Maximum expansion steps at each time step (default: 1).
        prefix_alpha: Maximum prefix length in prefix search (default: 1).
        expansion_gamma: Log probability difference for prune-by-value method (default: 2.3).
        expansion_beta: Additional candidates for expanded hypotheses selection (default: 2).
        score_norm: Normalize final scores by length (default: True).
        score_norm_during: Normalize scores by length during search (default: False).
        nbest: Number of final hypotheses to return (default: 1).
        penalty: Penalty applied to scores (default: 0.0).
        token_list: Optional list of tokens for decoding output.
        hold_n: Number of tokens to hold for incremental decoding (default: 0).

    Examples:
        >>> decoder = MyDecoder()
        >>> joint_network = MyJointNetwork()
        >>> beam_search = BeamSearchTransducerStreaming(
        ...     decoder=decoder,
        ...     joint_network=joint_network,
        ...     beam_size=5,
        ...     lm=my_language_model
        ... )
        >>> enc_out = torch.randn(10, decoder.dunits)  # Example encoder output
        >>> hypotheses = beam_search(enc_out)

    Raises:
        NotImplementedError: If an unsupported search type or language model is provided.

    Note:
        The `search_type` can be one of the following:
            - "default"
            - "greedy"
            - "tsd"
            - "alsd"
            - "nsc"
            - "maes"
        Each type has its own specific behavior and performance characteristics.
    """

    def __init__(
        self,
        decoder: AbsDecoder,
        joint_network: JointNetwork,
        beam_size: int,
        lm: torch.nn.Module = None,
        lm_weight: float = 0.1,
        search_type: str = "default",
        max_sym_exp: int = 2,
        u_max: int = 50,
        nstep: int = 1,
        prefix_alpha: int = 1,
        expansion_gamma: int = 2.3,
        expansion_beta: int = 2,
        score_norm: bool = True,
        score_norm_during: bool = False,
        nbest: int = 1,
        penalty: float = 0.0,
        token_list: Optional[List[str]] = None,
        hold_n: int = 0,
    ):
        """Initialize Transducer search module.

        Args:
            decoder: Decoder module.
            joint_network: Joint network module.
            beam_size: Beam size.
            lm: LM class.
            lm_weight: LM weight for soft fusion.
            search_type: Search algorithm to use during inference.
            max_sym_exp: Number of maximum symbol expansions at each time step. (TSD)
            u_max: Maximum output sequence length. (ALSD)
            nstep: Number of maximum expansion steps at each time step. (NSC/mAES)
            prefix_alpha: Maximum prefix length in prefix search. (NSC/mAES)
            expansion_beta:
              Number of additional candidates for expanded hypotheses selection. (mAES)
            expansion_gamma: Allowed logp difference for prune-by-value method. (mAES)
            score_norm: Normalize final scores by length. ("default")
            score_norm_during:
              Normalize scores by length during search. (default, TSD, ALSD)
            nbest: Number of final hypothesis.

        """
        self.decoder = decoder
        self.joint_network = joint_network

        self.beam_size = beam_size
        self.hidden_size = decoder.dunits
        self.vocab_size = decoder.odim

        self.sos = self.vocab_size - 1
        self.token_list = token_list

        self.blank_id = decoder.blank_id

        self.penalty = penalty

        if self.beam_size <= 1:
            self.search_algorithm = self.greedy_search
        elif search_type == "default":
            self.search_algorithm = self.default_beam_search
        elif search_type == "default2":
            self.search_algorithm = self.default_beam_search_v2
        elif search_type == "tsd":
            if isinstance(lm, TransformerLM):
                raise NotImplementedError

            self.max_sym_exp = max_sym_exp

            self.search_algorithm = self.time_sync_decoding
        elif search_type == "tsd2":
            if isinstance(lm, TransformerLM):
                raise NotImplementedError

            self.max_sym_exp = max_sym_exp

            self.search_algorithm = self.time_sync_decoding_v2
        elif search_type == "alsd":
            if isinstance(lm, TransformerLM):
                raise NotImplementedError

            self.u_max = u_max

            self.search_algorithm = self.align_length_sync_decoding
        elif search_type == "alsd2":
            if isinstance(lm, TransformerLM):
                raise NotImplementedError

            self.u_max = u_max

            self.search_algorithm = self.align_length_sync_decoding_v2
        elif search_type == "nsc":
            if isinstance(lm, TransformerLM):
                raise NotImplementedError

            self.nstep = nstep
            self.prefix_alpha = prefix_alpha

            self.search_algorithm = self.nsc_beam_search
        elif search_type == "maes":
            if isinstance(lm, TransformerLM):
                raise NotImplementedError

            self.nstep = nstep if nstep > 1 else 2
            self.prefix_alpha = prefix_alpha
            self.expansion_gamma = expansion_gamma

            assert self.vocab_size >= beam_size + expansion_beta, (
                "beam_size (%d) + expansion_beta (%d) "
                "should be smaller or equal to vocabulary size (%d)."
                % (beam_size, expansion_beta, self.vocab_size)
            )
            self.max_candidates = beam_size + expansion_beta

            self.search_algorithm = self.modified_adaptive_expansion_search
        else:
            raise NotImplementedError

        self.use_lm = lm is not None
        self.lm = lm
        self.lm_weight = lm_weight

        self.score_norm = score_norm
        self.score_norm_during = score_norm_during
        self.nbest = nbest

        self.hold_n = hold_n
        self.reset()

    def __call__(
        self,
        enc_out: torch.Tensor,
        start_idx: int = 0,
        is_final: bool = False,
        incremental_decode: bool = False,
    ) -> Union[List[Hypothesis], List[ExtendedHypothesis]]:
        """Perform beam search.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        self.decoder.set_device(enc_out.device)

        nbest_hyps = self.search_algorithm(enc_out[start_idx:])

        if incremental_decode:
            # prune hyps not containing top hyp as a prefix
            if len(nbest_hyps[0].yseq) > self.hold_n and not is_final:
                inc = nbest_hyps[0].yseq[: len(nbest_hyps[0].yseq) - self.hold_n]
                logging.debug(
                    "top hyp: "
                    + "".join([self.token_list[x] for x in nbest_hyps[0].yseq])
                )
                logging.debug(
                    "top hyp hold_n: " + "".join([self.token_list[x] for x in inc])
                )
            else:
                inc = nbest_hyps[0].yseq
            new_B = [nbest_hyps[0]]
            for h in nbest_hyps[1:]:
                if len(h.yseq) <= len(inc):
                    continue
                keep = True
                for i in range(len(inc)):
                    if inc[i] != h.yseq[i]:
                        keep = False
                        break
                if keep:
                    new_B.append(h)
            logging.debug(f"hyps after inc pruning: {len(new_B)}")

            self.B = new_B

            ret = [
                Hypothesis2(
                    yseq=torch.tensor(list(inc) + [self.sos]), score=nbest_hyps[0].score
                )
            ]
        else:
            ret = [
                Hypothesis2(yseq=torch.tensor(h.yseq + [self.sos]), score=h.score)
                for h in nbest_hyps
            ]

        if is_final:
            self.reset()

        return ret

    def reset(self):
        """
        Reset the beam search state.

        This method initializes the beam search state by resetting the
        hypotheses and beam states. It sets the initial hypotheses with
        the blank token and a score of zero, and initializes the decoder's
        state for the beam size.

        Attributes:
            beam (int): The effective beam size, constrained by the
                vocabulary size.
            beam_state: The initial state of the decoder for the beam.
            B (List[Hypothesis]): The list of hypotheses to keep track of
                during the search process.
            cache (dict): A cache for storing intermediate results.

        Example:
            >>> beam_search_transducer = BeamSearchTransducerStreaming(...)
            >>> beam_search_transducer.reset()
            >>> print(beam_search_transducer.B)  # Should show initial hypotheses

        Note:
            This method is called at the beginning of each decoding
            session and after final results are obtained to prepare
            for the next decoding task.
        """
        beam = min(self.beam_size, self.vocab_size)
        self.beam_state = self.decoder.init_state(beam)
        self.B = [
            Hypothesis(
                yseq=[self.blank_id],
                score=0.0,
                dec_state=self.decoder.select_state(self.beam_state, 0),
            )
        ]
        self.cache = {}

    def sort_nbest(
        self, hyps: Union[List[Hypothesis], List[ExtendedHypothesis]]
    ) -> Union[List[Hypothesis], List[ExtendedHypothesis]]:
        """
        Sort hypotheses by score or score given sequence length.

        This method sorts the provided hypotheses based on their scores.
        The sorting can be done in two ways:
        - By the raw score (if `score_norm` is False).
        - By the score normalized by the length of the sequence (if `score_norm` is True).

        Args:
            hyps: A list of hypotheses to be sorted. The hypotheses can be of
                type `Hypothesis` or `ExtendedHypothesis`.

        Returns:
            A list of the top `nbest` sorted hypotheses.

        Examples:
            >>> beam_search = BeamSearchTransducerStreaming(...)
            >>> hyps = [Hypothesis(score=5.0, yseq=[1, 2], dec_state=...),
            ...         Hypothesis(score=3.0, yseq=[1, 3], dec_state=...)]
            >>> sorted_hyps = beam_search.sort_nbest(hyps)
            >>> print(sorted_hyps)
            [Hypothesis(score=5.0, yseq=[1, 2], dec_state=...),
             Hypothesis(score=3.0, yseq=[1, 3], dec_state=...)]

        Note:
            The sorting will keep only the top `nbest` hypotheses in the
            returned list.
        """
        if self.score_norm:
            hyps.sort(key=lambda x: x.score / len(x.yseq), reverse=True)
        else:
            hyps.sort(key=lambda x: x.score, reverse=True)

        return hyps[: self.nbest]

    def prefix_search(
        self, hyps: List[ExtendedHypothesis], enc_out_t: torch.Tensor
    ) -> List[ExtendedHypothesis]:
        """
        Prefix search for NSC and mAES strategies.

        This method performs a prefix search among the given hypotheses to
        update their scores based on the encoder output at the current
        time step. The search is designed to be efficient by leveraging
        the prefix nature of the hypotheses, allowing for effective
        pruning and score adjustment.

        Args:
            hyps: A list of ExtendedHypothesis objects representing the
                current hypotheses.
            enc_out_t: The encoder output tensor for the current time step,
                shaped (D_enc,).

        Returns:
            A list of ExtendedHypothesis objects with updated scores.

        Examples:
            >>> beam_search = BeamSearchTransducerStreaming(...)
            >>> current_hyps = [...]  # List of ExtendedHypothesis objects
            >>> enc_out_t = torch.tensor([...])  # Current encoder output
            >>> updated_hyps = beam_search.prefix_search(current_hyps, enc_out_t)

        Note:
            This implementation is based on the methodology described in
            the paper: https://arxiv.org/pdf/1211.3711.pdf
        """
        for j, hyp_j in enumerate(hyps[:-1]):
            for hyp_i in hyps[(j + 1) :]:
                curr_id = len(hyp_j.yseq)
                pref_id = len(hyp_i.yseq)

                if (
                    is_prefix(hyp_j.yseq, hyp_i.yseq)
                    and (curr_id - pref_id) <= self.prefix_alpha
                ):
                    logp = torch.log_softmax(
                        self.joint_network(enc_out_t, hyp_i.dec_out[-1]),
                        dim=-1,
                    )

                    curr_score = hyp_i.score + float(logp[hyp_j.yseq[pref_id]])

                    for k in range(pref_id, (curr_id - 1)):
                        logp = torch.log_softmax(
                            self.joint_network(enc_out_t, hyp_j.dec_out[k]),
                            dim=-1,
                        )

                        curr_score += float(logp[hyp_j.yseq[k + 1]])

                    hyp_j.score = np.logaddexp(hyp_j.score, curr_score)

        return hyps

    def greedy_search(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """
        Greedy search implementation for sequence decoding.

        This method performs a greedy search on the encoder output to find the
        most likely sequence of hypotheses based on the given transducer model.
        The algorithm iteratively selects the token with the highest probability
        at each time step until the end of the sequence is reached.

        Args:
            enc_out: A tensor representing the encoder output sequence of shape
                (T, D_enc), where T is the number of time steps and D_enc is the
                dimension of the encoder output.

        Returns:
            List[Hypothesis]: A list containing the single best hypothesis, which
                includes the score and the sequence of tokens predicted.

        Examples:
            >>> enc_output = torch.randn(10, 128)  # Example encoder output
            >>> transducer = BeamSearchTransducerStreaming(...)  # Initialized object
            >>> best_hypothesis = transducer.greedy_search(enc_output)
            >>> print(best_hypothesis[0].yseq)  # Output the predicted sequence

        Note:
            This method assumes that the decoder has been properly initialized
            and that the encoder output is valid. The output will always contain
            a single hypothesis, which represents the greedy choice made at each
            decoding step.
        """
        dec_state = self.decoder.init_state(1)

        hyp = Hypothesis(score=0.0, yseq=[self.blank_id], dec_state=dec_state)
        cache = {}

        dec_out, state, _ = self.decoder.score(hyp, cache)

        for enc_out_t in enc_out:
            logp = torch.log_softmax(
                self.joint_network(enc_out_t, dec_out),
                dim=-1,
            )
            top_logp, pred = torch.max(logp, dim=-1)

            if pred != self.blank_id:
                hyp.yseq.append(int(pred))
                hyp.score += float(top_logp)

                hyp.dec_state = state

                dec_out, state, _ = self.decoder.score(hyp, cache)

        return [hyp]

    def default_beam_search(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """
        Beam search implementation.

        This method performs a standard beam search decoding algorithm for
        transducer models, where the search explores multiple possible
        hypotheses at each decoding step. The algorithm retains the top
        scoring hypotheses for further expansion while discarding less
        promising candidates.

        The implementation is inspired by the method described in the paper
        "Connectionist Temporal Classification: Labelling Unsegmented
        Sequence Data with Recurrent Neural Networks" (https://arxiv.org/pdf/1211.3711.pdf).

        Args:
            enc_out: Encoder output sequence of shape (T, D), where T is
                the number of time steps and D is the dimensionality of the
                encoder output.

        Returns:
            nbest_hyps: A list of N-best hypotheses generated from the beam
                search process, sorted by their scores in descending order.

        Examples:
            >>> model = BeamSearchTransducerStreaming(decoder, joint_network,
            ...                                       beam_size=5)
            >>> encoder_output = torch.rand(10, model.hidden_size)
            >>> results = model.default_beam_search(encoder_output)
            >>> for hyp in results:
            ...     print(hyp.yseq, hyp.score)
        """
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))

        dec_state = self.decoder.init_state(1)

        kept_hyps = [Hypothesis(score=0.0, yseq=[self.blank_id], dec_state=dec_state)]
        cache = {}
        cache_lm = {}

        for enc_out_t in enc_out:
            hyps = kept_hyps
            kept_hyps = []

            if self.token_list is not None:
                logging.debug(
                    "\n"
                    + "\n".join(
                        [
                            "hypo: "
                            + "".join([self.token_list[x] for x in hyp.yseq[1:]])
                            + f", score: {round(float(hyp.score), 2)}"
                            for hyp in sorted(hyps, key=lambda x: x.score, reverse=True)
                        ]
                    )
                )

            while True:
                if self.score_norm_during:
                    max_hyp = max(hyps, key=lambda x: x.score / len(x.yseq))
                else:
                    max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                dec_out, state, lm_tokens = self.decoder.score(max_hyp, cache)

                logp = torch.log_softmax(
                    self.joint_network(enc_out_t, dec_out),
                    dim=-1,
                )
                top_k = logp[1:].topk(beam_k, dim=-1)

                kept_hyps.append(
                    Hypothesis(
                        score=(max_hyp.score + float(logp[0:1])),
                        yseq=max_hyp.yseq[:],
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                    )
                )

                if self.use_lm:
                    if tuple(max_hyp.yseq) not in cache_lm:
                        lm_scores, lm_state = self.lm.score(
                            torch.LongTensor(
                                [self.sos] + max_hyp.yseq[1:],
                                device=self.decoder.device,
                            ),
                            max_hyp.lm_state,
                            None,
                        )
                        cache_lm[tuple(max_hyp.yseq)] = (lm_scores, lm_state)
                    else:
                        lm_scores, lm_state = cache_lm[tuple(max_hyp.yseq)]
                else:
                    lm_state = max_hyp.lm_state

                for logp, k in zip(*top_k):
                    score = max_hyp.score + float(logp)

                    if self.use_lm:
                        score += self.lm_weight * lm_scores[k + 1]

                    hyps.append(
                        Hypothesis(
                            score=score,
                            yseq=max_hyp.yseq[:] + [int(k + 1)],
                            dec_state=state,
                            lm_state=lm_state,
                        )
                    )

                if self.score_norm_during:
                    hyps_max = float(
                        max(hyps, key=lambda x: x.score / len(x.yseq)).score
                    )
                else:
                    hyps_max = float(max(hyps, key=lambda x: x.score).score)
                kept_most_prob = sorted(
                    [hyp for hyp in kept_hyps if hyp.score > hyps_max],
                    key=lambda x: x.score,
                )
                if len(kept_most_prob) >= beam:
                    kept_hyps = kept_most_prob
                    break

        return self.sort_nbest(kept_hyps)

    def time_sync_decoding(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """
        Time synchronous beam search implementation.

        This method performs a time-synchronous beam search decoding using
        the encoder output. It generates N-best hypotheses based on the
        provided encoder output sequence. The algorithm follows the
        principles outlined in the research paper:
        "End-to-End Speech Recognition with Transformer"
        (https://ieeexplore.ieee.org/document/9053040).

        Args:
            enc_out: A tensor representing the encoder output sequence
                with shape (T, D), where T is the number of time steps
                and D is the dimensionality of the output.

        Returns:
            List[Hypothesis]: A list containing the N-best hypotheses,
            where each hypothesis includes a sequence of predicted
            tokens and their associated scores.

        Examples:
            >>> encoder_output = torch.randn(10, 256)  # Example encoder output
            >>> decoder = BeamSearchTransducerStreaming(...)  # Initialize the decoder
            >>> hypotheses = decoder.time_sync_decoding(encoder_output)
            >>> for hyp in hypotheses:
            >>>     print(f'Score: {hyp.score}, Sequence: {hyp.yseq}')

        Note:
            The method supports language model integration if a language model
            is provided during the initialization of the decoder. The scoring
            incorporates language model scores based on the specified parameters.

        Raises:
            ValueError: If the encoder output tensor does not conform to the
            expected shape or dimensions.
        """
        beam = min(self.beam_size, self.vocab_size)

        beam_state = self.decoder.init_state(beam)

        B = [
            Hypothesis(
                yseq=[self.blank_id],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]
        cache = {}

        if self.use_lm:
            B[0].lm_state = self.lm.zero_state()

        for enc_out_t in enc_out:
            A = []
            C = B

            enc_out_t = enc_out_t.unsqueeze(0)

            for v in range(self.max_sym_exp):
                D = []

                beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    C,
                    beam_state,
                    cache,
                    self.use_lm,
                )

                beam_logp = torch.log_softmax(
                    self.joint_network(enc_out_t, beam_dec_out),
                    dim=-1,
                )
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

                if v < (self.max_sym_exp - 1):
                    if self.use_lm:
                        beam_lm_scores, beam_lm_states = self.lm.batch_score(
                            beam_lm_tokens, [c.lm_state for c in C], None
                        )

                    for i, hyp in enumerate(C):
                        for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                            new_hyp = Hypothesis(
                                score=(hyp.score + float(logp)),
                                yseq=(hyp.yseq + [int(k)]),
                                dec_state=self.decoder.select_state(beam_state, i),
                                lm_state=hyp.lm_state,
                            )

                            if self.use_lm:
                                new_hyp.score += self.lm_weight * beam_lm_scores[i, k]
                                new_hyp.lm_state = beam_lm_states[i]

                            D.append(new_hyp)

                if self.score_norm_during:
                    C = sorted(D, key=lambda x: x.score / len(x.yseq), reverse=True)[
                        :beam
                    ]
                else:
                    C = sorted(D, key=lambda x: x.score, reverse=True)[:beam]

            if self.score_norm_during:
                B = sorted(A, key=lambda x: x.score / len(x.yseq), reverse=True)[:beam]
            else:
                B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]

        return self.sort_nbest(B)

    def align_length_sync_decoding(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """
        Alignment-length synchronous beam search implementation.

        This method implements an alignment-length synchronous beam search
        algorithm based on the paper available at
        https://ieeexplore.ieee.org/document/9053040. The algorithm is
        designed to handle decoding in a way that aligns the output sequence
        length with the input sequence length, thereby maintaining a
        synchronous relationship during the decoding process.

        Args:
            enc_out: Encoder output sequences. (T, D), where T is the
                      number of time steps and D is the dimension of the
                      encoder output.

        Returns:
            List[Hypothesis]: N-best hypotheses, sorted by score, where each
                              hypothesis contains a sequence of predicted
                              tokens and the associated score.

        Examples:
            >>> model = BeamSearchTransducerStreaming(...)
            >>> enc_output = torch.randn(100, model.hidden_size)
            >>> nbest_hyps = model.align_length_sync_decoding(enc_output)
            >>> for hyp in nbest_hyps:
            ...     print(hyp.yseq, hyp.score)

        Note:
            The method utilizes a maximum output length (u_max) to control
            the number of tokens that can be generated, and it processes the
            encoder outputs in a way that aligns with the length of the
            predicted sequences.
        """
        beam = min(self.beam_size, self.vocab_size)

        t_max = int(enc_out.size(0))
        u_max = min(self.u_max, (t_max - 1))

        beam_state = self.decoder.init_state(beam)

        B = [
            Hypothesis(
                yseq=[self.blank_id],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]
        final = []
        cache = {}

        if self.use_lm:
            B[0].lm_state = self.lm.zero_state()

        for i in range(t_max + u_max):
            A = []

            B_ = []
            B_enc_out = []
            for hyp in B:
                u = len(hyp.yseq) - 1
                t = i - u

                if t > (t_max - 1):
                    continue

                B_.append(hyp)
                B_enc_out.append((t, enc_out[t]))

            if B_:
                beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    B_,
                    beam_state,
                    cache,
                    self.use_lm,
                )

                beam_enc_out = torch.stack([x[1] for x in B_enc_out])

                beam_logp = torch.log_softmax(
                    self.joint_network(beam_enc_out, beam_dec_out),
                    dim=-1,
                )
                beam_topk = beam_logp[:, 1:].topk(beam, dim=-1)

                if self.use_lm:
                    beam_lm_scores, beam_lm_states = self.lm.batch_score(
                        beam_lm_tokens,
                        [b.lm_state for b in B_],
                        None,
                    )

                for i, hyp in enumerate(B_):
                    new_hyp = Hypothesis(
                        score=(hyp.score + float(beam_logp[i, 0])),
                        yseq=hyp.yseq[:],
                        dec_state=hyp.dec_state,
                        lm_state=hyp.lm_state,
                    )

                    A.append(new_hyp)

                    if B_enc_out[i][0] == (t_max - 1):
                        final.append(new_hyp)

                    for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                        new_hyp = Hypothesis(
                            score=(hyp.score + float(logp)),
                            yseq=(hyp.yseq[:] + [int(k)]),
                            dec_state=self.decoder.select_state(beam_state, i),
                            lm_state=hyp.lm_state,
                        )

                        if self.use_lm:
                            new_hyp.score += self.lm_weight * beam_lm_scores[i, k]
                            new_hyp.lm_state = beam_lm_states[i]

                        A.append(new_hyp)

                if self.score_norm_during:
                    B = sorted(A, key=lambda x: x.score / len(x.yseq), reverse=True)[
                        :beam
                    ]
                else:
                    B = sorted(A, key=lambda x: x.score, reverse=True)[:beam]
                B = recombine_hyps(B)

        if final:
            return self.sort_nbest(final)
        else:
            return B

    def nsc_beam_search(self, enc_out: torch.Tensor) -> List[ExtendedHypothesis]:
        """
        N-step constrained beam search implementation.

        This method performs N-step constrained beam search based on the input
        encoder output sequence. It is designed to efficiently search for the
        best hypotheses by constraining the number of steps and utilizing
        prefix search techniques. This algorithm is based on and modified
        from the paper "https://arxiv.org/pdf/2002.03577.pdf". For any usage
        outside ESPnet, please reference ESPnet (b-flo, PR #2444) until further
        modifications are made.

        Args:
            enc_out: Encoder output sequence. Shape is (T, D_enc), where T is the
                    length of the sequence and D_enc is the dimensionality of
                    the encoder output.

        Returns:
            List[ExtendedHypothesis]: A list of N-best hypotheses sorted by score.
                                    Each hypothesis includes the score, the
                                    sequence of tokens generated, the decoder
                                    state, and any associated language model
                                    scores.

        Examples:
            >>> # Assuming enc_out is a tensor of appropriate shape
            >>> decoder = ...  # Initialize your decoder
            >>> joint_network = ...  # Initialize your joint network
            >>> beam_search = BeamSearchTransducerStreaming(decoder, joint_network, beam_size=5)
            >>> nbest_hyps = beam_search.nsc_beam_search(enc_out)
            >>> for hyp in nbest_hyps:
            >>>     print(hyp.yseq, hyp.score)

        Note:
            This implementation may require a language model (LM) for scoring,
            which can be provided during initialization of the BeamSearchTransducerStreaming
            class.

        Raises:
            NotImplementedError: If an unsupported language model type is used
                                during initialization.
        """
        beam = min(self.beam_size, self.vocab_size)
        beam_k = min(beam, (self.vocab_size - 1))

        beam_state = self.decoder.init_state(beam)

        init_tokens = [
            ExtendedHypothesis(
                yseq=[self.blank_id],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]

        cache = {}

        beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
            init_tokens,
            beam_state,
            cache,
            self.use_lm,
        )

        state = self.decoder.select_state(beam_state, 0)

        if self.use_lm:
            beam_lm_scores, beam_lm_states = self.lm.batch_score(
                beam_lm_tokens,
                [i.lm_state for i in init_tokens],
                None,
            )
            lm_state = beam_lm_states[0]
            lm_scores = beam_lm_scores[0]
        else:
            lm_state = None
            lm_scores = None

        kept_hyps = [
            ExtendedHypothesis(
                yseq=[self.blank_id],
                score=0.0,
                dec_state=state,
                dec_out=[beam_dec_out[0]],
                lm_state=lm_state,
                lm_scores=lm_scores,
            )
        ]

        for enc_out_t in enc_out:
            hyps = self.prefix_search(
                sorted(kept_hyps, key=lambda x: len(x.yseq), reverse=True),
                enc_out_t,
            )
            kept_hyps = []

            beam_enc_out = enc_out_t.unsqueeze(0)

            S = []
            V = []
            for n in range(self.nstep):
                beam_dec_out = torch.stack([hyp.dec_out[-1] for hyp in hyps])

                beam_logp = torch.log_softmax(
                    self.joint_network(beam_enc_out, beam_dec_out),
                    dim=-1,
                )
                beam_topk = beam_logp[:, 1:].topk(beam_k, dim=-1)

                for i, hyp in enumerate(hyps):
                    S.append(
                        ExtendedHypothesis(
                            yseq=hyp.yseq[:],
                            score=hyp.score + float(beam_logp[i, 0:1]),
                            dec_out=hyp.dec_out[:],
                            dec_state=hyp.dec_state,
                            lm_state=hyp.lm_state,
                            lm_scores=hyp.lm_scores,
                        )
                    )

                    for logp, k in zip(beam_topk[0][i], beam_topk[1][i] + 1):
                        score = hyp.score + float(logp)

                        if self.use_lm:
                            score += self.lm_weight * float(hyp.lm_scores[k])

                        V.append(
                            ExtendedHypothesis(
                                yseq=hyp.yseq[:] + [int(k)],
                                score=score,
                                dec_out=hyp.dec_out[:],
                                dec_state=hyp.dec_state,
                                lm_state=hyp.lm_state,
                                lm_scores=hyp.lm_scores,
                            )
                        )

                V.sort(key=lambda x: x.score, reverse=True)
                V = subtract(V, hyps)[:beam]

                beam_state = self.decoder.create_batch_states(
                    beam_state,
                    [v.dec_state for v in V],
                    [v.yseq for v in V],
                )
                beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
                    V,
                    beam_state,
                    cache,
                    self.use_lm,
                )

                if self.use_lm:
                    beam_lm_scores, beam_lm_states = self.lm.batch_score(
                        beam_lm_tokens, [v.lm_state for v in V], None
                    )

                if n < (self.nstep - 1):
                    for i, v in enumerate(V):
                        v.dec_out.append(beam_dec_out[i])

                        v.dec_state = self.decoder.select_state(beam_state, i)

                        if self.use_lm:
                            v.lm_state = beam_lm_states[i]
                            v.lm_scores = beam_lm_scores[i]

                    hyps = V[:]
                else:
                    beam_logp = torch.log_softmax(
                        self.joint_network(beam_enc_out, beam_dec_out),
                        dim=-1,
                    )

                    for i, v in enumerate(V):
                        if self.nstep != 1:
                            v.score += float(beam_logp[i, 0])

                        v.dec_out.append(beam_dec_out[i])

                        v.dec_state = self.decoder.select_state(beam_state, i)

                        if self.use_lm:
                            v.lm_state = beam_lm_states[i]
                            v.lm_scores = beam_lm_scores[i]

            kept_hyps = sorted((S + V), key=lambda x: x.score, reverse=True)[:beam]

        return self.sort_nbest(kept_hyps)

    def modified_adaptive_expansion_search(
        self, enc_out: torch.Tensor
    ) -> List[ExtendedHypothesis]:
        """
        Perform the modified Adaptive Expansion Search (mAES) for decoding.

        This method implements the modified Adaptive Expansion Search
        algorithm, which is based on the work presented in
        https://ieeexplore.ieee.org/document/9250505 and incorporates
        elements from the N-step Constrained beam search (NSC).

        Args:
            enc_out: Encoder output sequence. Shape (T, D_enc), where T is
                the number of time steps and D_enc is the dimension of the
                encoder output.

        Returns:
            List[ExtendedHypothesis]: A list of the N-best hypotheses
            generated by the search algorithm, each represented as an
            ExtendedHypothesis instance.

        Examples:
            >>> enc_out = torch.randn(10, 256)  # Example encoder output
            >>> beam_search = BeamSearchTransducerStreaming(...)  # Initialize
            >>> nbest_hyps = beam_search.modified_adaptive_expansion_search(enc_out)
            >>> for hyp in nbest_hyps:
            >>>     print(hyp.yseq, hyp.score)

        Note:
            This method is designed to be used within a beam search
            framework and relies on a well-defined decoder and joint
            network to compute scores and states for hypotheses.
        """
        beam = min(self.beam_size, self.vocab_size)
        beam_state = self.decoder.init_state(beam)

        init_tokens = [
            ExtendedHypothesis(
                yseq=[self.blank_id],
                score=0.0,
                dec_state=self.decoder.select_state(beam_state, 0),
            )
        ]

        cache = {}

        beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
            init_tokens,
            beam_state,
            cache,
            self.use_lm,
        )

        state = self.decoder.select_state(beam_state, 0)

        if self.use_lm:
            beam_lm_scores, beam_lm_states = self.lm.batch_score(
                beam_lm_tokens, [i.lm_state for i in init_tokens], None
            )

            lm_state = beam_lm_states[0]
            lm_scores = beam_lm_scores[0]
        else:
            lm_state = None
            lm_scores = None

        kept_hyps = [
            ExtendedHypothesis(
                yseq=[self.blank_id],
                score=0.0,
                dec_state=state,
                dec_out=[beam_dec_out[0]],
                lm_state=lm_state,
                lm_scores=lm_scores,
            )
        ]

        for enc_out_t in enc_out:
            hyps = self.prefix_search(
                sorted(kept_hyps, key=lambda x: len(x.yseq), reverse=True),
                enc_out_t,
            )
            kept_hyps = []

            beam_enc_out = enc_out_t.unsqueeze(0)

            list_b = []
            duplication_check = [hyp.yseq for hyp in hyps]

            for n in range(self.nstep):
                beam_dec_out = torch.stack([h.dec_out[-1] for h in hyps])

                beam_logp, beam_idx = torch.log_softmax(
                    self.joint_network(beam_enc_out, beam_dec_out),
                    dim=-1,
                ).topk(self.max_candidates, dim=-1)

                k_expansions = select_k_expansions(
                    hyps,
                    beam_idx,
                    beam_logp,
                    self.expansion_gamma,
                )

                list_exp = []
                for i, hyp in enumerate(hyps):
                    for k, new_score in k_expansions[i]:
                        new_hyp = ExtendedHypothesis(
                            yseq=hyp.yseq[:],
                            score=new_score,
                            dec_out=hyp.dec_out[:],
                            dec_state=hyp.dec_state,
                            lm_state=hyp.lm_state,
                            lm_scores=hyp.lm_scores,
                        )

                        if k == 0:
                            list_b.append(new_hyp)
                        else:
                            if new_hyp.yseq + [int(k)] not in duplication_check:
                                new_hyp.yseq.append(int(k))

                                if self.use_lm:
                                    new_hyp.score += self.lm_weight * float(
                                        hyp.lm_scores[k]
                                    )

                                list_exp.append(new_hyp)

                if not list_exp:
                    kept_hyps = sorted(list_b, key=lambda x: x.score, reverse=True)[
                        :beam
                    ]

                    break
                else:
                    beam_state = self.decoder.create_batch_states(
                        beam_state,
                        [hyp.dec_state for hyp in list_exp],
                        [hyp.yseq for hyp in list_exp],
                    )

                    beam_dec_out, beam_state, beam_lm_tokens = self.decoder.batch_score(
                        list_exp,
                        beam_state,
                        cache,
                        self.use_lm,
                    )

                    if self.use_lm:
                        beam_lm_scores, beam_lm_states = self.lm.batch_score(
                            beam_lm_tokens, [k.lm_state for k in list_exp], None
                        )

                    if n < (self.nstep - 1):
                        for i, hyp in enumerate(list_exp):
                            hyp.dec_out.append(beam_dec_out[i])
                            hyp.dec_state = self.decoder.select_state(beam_state, i)

                            if self.use_lm:
                                hyp.lm_state = beam_lm_states[i]
                                hyp.lm_scores = beam_lm_scores[i]

                        hyps = list_exp[:]
                    else:
                        beam_logp = torch.log_softmax(
                            self.joint_network(beam_enc_out, beam_dec_out),
                            dim=-1,
                        )

                        for i, hyp in enumerate(list_exp):
                            hyp.score += float(beam_logp[i, 0])

                            hyp.dec_out.append(beam_dec_out[i])
                            hyp.dec_state = self.decoder.select_state(beam_state, i)

                            if self.use_lm:
                                hyp.lm_states = beam_lm_states[i]
                                hyp.lm_scores = beam_lm_scores[i]

                        kept_hyps = sorted(
                            list_b + list_exp, key=lambda x: x.score, reverse=True
                        )[:beam]

        return self.sort_nbest(kept_hyps)
