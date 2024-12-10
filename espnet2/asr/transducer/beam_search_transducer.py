"""Search algorithms for Transducer models."""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr_transducer.joint_network import JointNetwork
from espnet2.lm.transformer_lm import TransformerLM
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

    This class represents a single hypothesis during the search process of 
    Transducer models, encapsulating the score, output sequence, and 
    decoder state.

    Attributes:
        score (float): The score of the hypothesis, representing its 
            likelihood.
        yseq (List[int]): The output sequence represented as a list of 
            integers (token indices).
        dec_state (Union[Tuple[torch.Tensor, Optional[torch.Tensor]], 
            List[Optional[torch.Tensor]], torch.Tensor]): The state of the 
            decoder corresponding to the current hypothesis. This can be a 
            tuple of tensors or a list of tensors, depending on the decoder 
            implementation.
        lm_state (Union[Dict[str, Any], List[Any]], optional): The state of 
            the language model, if applicable. Defaults to None.

    Examples:
        >>> hyp = Hypothesis(score=0.0, yseq=[1, 2, 3], dec_state=(torch.tensor([0.1]),))
        >>> print(hyp.score)
        0.0
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

    This class extends the basic Hypothesis structure to include additional
    information specific to Non-Standard Constrained (NSC) beam search and
    modified Adaptive Expansion Search (mAES) algorithms. It includes
    decoder outputs and language model scores to enhance the search
    capabilities.

    Attributes:
        dec_out (List[torch.Tensor]): A list of decoder output tensors for each
            step in the hypothesis generation.
        lm_scores (torch.Tensor): Tensor holding language model scores for
            the tokens in the hypothesis.

    Examples:
        >>> # Create an instance of ExtendedHypothesis
        >>> hypothesis = ExtendedHypothesis(
        ...     score=10.0,
        ...     yseq=[1, 2, 3],
        ...     dec_state=(torch.tensor([0.1]), None),
        ...     dec_out=[torch.tensor([0.2]), torch.tensor([0.3])],
        ...     lm_scores=torch.tensor([0.5, 0.6])
        ... )
        >>> print(hypothesis)
        ExtendedHypothesis(score=10.0, yseq=[1, 2, 3], ...)

    Note:
        The `dec_out` and `lm_scores` attributes are particularly useful in
        contexts where multiple hypotheses need to be evaluated based on both
        decoder output and language model contributions.

    Todo:
        - Implement methods for manipulating and accessing the `dec_out` and
          `lm_scores` attributes more efficiently.
    """

    dec_out: List[torch.Tensor] = None
    lm_scores: torch.Tensor = None


class BeamSearchTransducer:
    """
    Beam search implementation for Transducer models.

    This class implements various beam search algorithms for Transducer 
    models used in automatic speech recognition (ASR). It allows for 
    flexible decoding strategies such as greedy search, beam search, 
    and more advanced techniques like N-step constrained search and 
    modified adaptive expansion search.

    Attributes:
        decoder (AbsDecoder): The decoder module used for generating 
            predictions.
        joint_network (JointNetwork): The joint network module used 
            for combining encoder and decoder outputs.
        beam_size (int): The size of the beam for beam search.
        lm (torch.nn.Module, optional): Language model for soft fusion.
        lm_weight (float): Weight for the language model during decoding.
        search_type (str): The search algorithm to use during inference.
        max_sym_exp (int): Maximum number of symbol expansions at each 
            time step.
        u_max (int): Maximum output sequence length.
        nstep (int): Number of maximum expansion steps at each time 
            step.
        prefix_alpha (int): Maximum prefix length in prefix search.
        expansion_gamma (int): Allowed log probability difference for 
            pruning.
        expansion_beta (int): Number of additional candidates for 
            expanded hypotheses selection.
        multi_blank_durations (List[int]): The duration of each blank 
            token.
        multi_blank_indices (List[int]): The index of each blank token 
            in token_list.
        score_norm (bool): Normalize final scores by length.
        score_norm_during (bool): Normalize scores by length during 
            search.
        nbest (int): Number of final hypotheses.
        token_list (List[str], optional): List of tokens used in 
            decoding.

    Args:
        decoder (AbsDecoder): Decoder module.
        joint_network (JointNetwork): Joint network module.
        beam_size (int): Beam size.
        lm (torch.nn.Module, optional): Language model class.
        lm_weight (float): Language model weight for soft fusion.
        search_type (str): Search algorithm to use during inference.
        max_sym_exp (int): Maximum symbol expansions at each time step.
        u_max (int): Maximum output sequence length.
        nstep (int): Maximum expansion steps at each time step.
        prefix_alpha (int): Maximum prefix length in prefix search.
        expansion_beta (int): Additional candidates for expanded 
            hypotheses.
        expansion_gamma (int): Log probability difference for pruning.
        multi_blank_durations (List[int]): Duration of each blank token.
        multi_blank_indices (List[int]): Index of each blank token.
        score_norm (bool): Normalize final scores by length.
        score_norm_during (bool): Normalize scores by length during search.
        nbest (int): Number of final hypotheses.
        token_list (List[str], optional): List of tokens.

    Returns:
        List[Hypothesis] or List[ExtendedHypothesis]: N-best decoding 
        results, depending on the search type used.

    Examples:
        >>> beam_search = BeamSearchTransducer(decoder, joint_network, 
        ...                                     beam_size=5)
        >>> enc_out = torch.randn(10, 256)  # Example encoder output
        >>> nbest_hyps = beam_search(enc_out)
        >>> for hyp in nbest_hyps:
        ...     print(hyp.yseq, hyp.score)

    Note:
        Ensure that the chosen search_type is compatible with the 
        provided language model, if any.

    Todo:
        - Implement additional search algorithms as required.
        - Optimize performance for large beam sizes.

    Raises:
        NotImplementedError: If an unsupported search type is provided.
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
        multi_blank_durations: List[int] = [],
        multi_blank_indices: List[int] = [],
        score_norm: bool = True,
        score_norm_during: bool = False,
        nbest: int = 1,
        token_list: Optional[List[str]] = None,
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
            multi_blank_durations: The duration of each blank token. (MBG)
            multi_blank_indices: The index of each blank token in token_list. (MBG)
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

        if search_type == "mbg":
            self.beam_size = 1
            self.multi_blank_durations = multi_blank_durations
            self.multi_blank_indices = multi_blank_indices
            self.search_algorithm = self.multi_blank_greedy_search

        elif self.beam_size <= 1:
            self.search_algorithm = self.greedy_search
        elif search_type == "default":
            self.search_algorithm = self.default_beam_search
        elif search_type == "tsd":
            if isinstance(lm, TransformerLM):
                raise NotImplementedError

            self.max_sym_exp = max_sym_exp

            self.search_algorithm = self.time_sync_decoding
        elif search_type == "alsd":
            if isinstance(lm, TransformerLM):
                raise NotImplementedError

            self.u_max = u_max

            self.search_algorithm = self.align_length_sync_decoding
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

        if self.use_lm and self.beam_size == 1:
            logging.warning("LM is provided but not used, since this is greedy search.")

        self.score_norm = score_norm
        self.score_norm_during = score_norm_during
        self.nbest = nbest

    def __call__(
        self, enc_out: torch.Tensor
    ) -> Union[List[Hypothesis], List[ExtendedHypothesis]]:
        """Perform beam search.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)

        Returns:
            nbest_hyps: N-best decoding results

        """
        self.decoder.set_device(enc_out.device)

        nbest_hyps = self.search_algorithm(enc_out)

        return nbest_hyps

    def sort_nbest(
        self, hyps: Union[List[Hypothesis], List[ExtendedHypothesis]]
    ) -> Union[List[Hypothesis], List[ExtendedHypothesis]]:
        """
        Sort hypotheses by score or score given sequence length.

        This method sorts the provided list of hypotheses based on their scores. 
        It can normalize the scores by the length of the sequences if specified.

        Args:
            hyps: A list of hypotheses to be sorted. This can be a list of 
                either `Hypothesis` or `ExtendedHypothesis` objects.

        Returns:
            A list of the top N best hypotheses, sorted in descending order of 
            score (or normalized score if applicable).

        Examples:
            >>> hyp1 = Hypothesis(score=10.0, yseq=[1, 2, 3], dec_state=None)
            >>> hyp2 = Hypothesis(score=12.0, yseq=[1, 2], dec_state=None)
            >>> hyp3 = Hypothesis(score=9.0, yseq=[1, 2, 3, 4], dec_state=None)
            >>> sorted_hyps = sort_nbest([hyp1, hyp2, hyp3])
            >>> print([hyp.score for hyp in sorted_hyps])
            [12.0, 10.0, 9.0]

        Note:
            If `score_norm` is set to `True`, the scores will be divided by the 
            length of the sequences before sorting.
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

        This method performs a prefix search over a list of hypotheses to 
        update their scores based on the current encoder output. It identifies 
        hypotheses that share a common prefix and adjusts their scores by 
        considering the log probabilities of extending these prefixes.

        The algorithm operates as follows:
        1. Iterate through each hypothesis and compare it with others.
        2. Check if one hypothesis is a prefix of another and if the 
           difference in their lengths is within the allowed prefix length.
        3. Calculate the log probability of extending the current hypothesis 
           using the joint network and update the scores accordingly.

        Args:
            hyps: A list of ExtendedHypothesis instances representing the 
                  current hypotheses.
            enc_out_t: The encoder output for the current time step.

        Returns:
            List[ExtendedHypothesis]: The updated list of hypotheses with 
            adjusted scores.

        Examples:
            >>> enc_out_t = torch.randn(1, 256)  # Example encoder output
            >>> hyp1 = ExtendedHypothesis(yseq=[1, 2], score=0.5, dec_out=[...])
            >>> hyp2 = ExtendedHypothesis(yseq=[1, 2, 3], score=0.6, dec_out=[...])
            >>> hyps = [hyp1, hyp2]
            >>> updated_hyps = prefix_search(hyps, enc_out_t)

        Note:
            This method is particularly useful in the context of 
            N-step constrained beam search (NSC) and modified adaptive 
            expansion search (mAES) strategies, where maintaining and 
            updating prefixes efficiently can lead to better decoding 
            performance.
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
        Greedy search implementation.

        This method performs a greedy search on the given encoder output sequence.
        It initializes a hypothesis with the blank token and iteratively updates
        the hypothesis by predicting the next token at each time step based on the
        encoder's output and the decoder's scores.

        Args:
            enc_out: Encoder output sequence of shape (T, D_enc), where T is the
                number of time steps and D_enc is the dimension of the encoder's
                output.

        Returns:
            List[Hypothesis]: A list containing the 1-best hypothesis found during
            the greedy search, which includes the score, predicted token sequence,
            and decoder state.

        Examples:
            >>> enc_out = torch.randn(10, 128)  # Example encoder output
            >>> beam_search_transducer = BeamSearchTransducer(...)
            >>> best_hypothesis = beam_search_transducer.greedy_search(enc_out)
            >>> print(best_hypothesis[0].yseq)  # Output the predicted sequence

        Note:
            This implementation assumes that the decoder has been initialized and
            is ready to perform scoring.
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

        This method performs beam search decoding for Transducer models. It
        utilizes the encoder output to generate N-best hypotheses. The search
        process is based on the principles outlined in the paper:
        "Sequence to Sequence Learning with Neural Networks" (https://arxiv.org/pdf/1211.3711.pdf).

        Args:
            enc_out: Encoder output sequence. The shape should be (T, D),
                      where T is the time dimension and D is the feature
                      dimension.

        Returns:
            List[Hypothesis]: A list of N-best hypotheses, each represented by
                              a Hypothesis object containing the score, 
                              output sequence, and decoder state.

        Examples:
            >>> enc_output = torch.randn(10, 256)  # Example encoder output
            >>> beam_search = BeamSearchTransducer(decoder, joint_network, beam_size=5)
            >>> nbest_hyps = beam_search.default_beam_search(enc_output)
            >>> for hyp in nbest_hyps:
            ...     print(hyp.yseq, hyp.score)

        Note:
            Ensure that the encoder output has been properly generated
            from the preceding model layers before calling this method.
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

        This method performs a time synchronous beam search over the encoder 
        output sequence. It maintains a beam of hypotheses and expands them 
        at each time step based on the scores from the joint network. The 
        search is guided by the beam size and considers the language model 
        if provided.

        This implementation is based on the work found in:
        https://ieeexplore.ieee.org/document/9053040

        Args:
            enc_out: Encoder output sequence. Shape (T, D), where T is the 
                      length of the sequence and D is the dimension of the 
                      encoder output.

        Returns:
            nbest_hyps: A list of N-best hypotheses sorted by score, each 
                         represented as an instance of the `Hypothesis` class.

        Examples:
            >>> # Example usage of the time_sync_decoding method
            >>> beam_search = BeamSearchTransducer(decoder, joint_network, beam_size=5)
            >>> encoder_output = torch.randn(10, 256)  # Example encoder output
            >>> results = beam_search.time_sync_decoding(encoder_output)
            >>> for hyp in results:
            >>>     print(hyp.yseq, hyp.score)

        Note:
            Ensure that the `decoder` and `joint_network` have been properly 
            initialized before calling this method.
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

        This method performs an alignment-length synchronous beam search for
        decoding sequences from an encoder output. The search iterates through 
        the time steps of the encoder output and maintains hypotheses that 
        align the length of the output sequences with the input encoder 
        output.

        The implementation is based on the research presented in:
        https://ieeexplore.ieee.org/document/9053040

        Args:
            enc_out: A tensor representing the encoder output sequences. 
                      Shape should be (T, D), where T is the number of time 
                      steps and D is the dimension of the encoder output.

        Returns:
            List[Hypothesis]: A list of N-best hypotheses, sorted by score.

        Examples:
            >>> enc_output = torch.rand(10, 256)  # Simulated encoder output
            >>> beam_search_transducer = BeamSearchTransducer(...)
            >>> nbest_hyps = beam_search_transducer.align_length_sync_decoding(enc_output)
            >>> for hyp in nbest_hyps:
            ...     print(hyp.yseq, hyp.score)

        Note:
            This method may produce a smaller number of final hypotheses 
            if no valid sequences are found that satisfy the length alignment 
            criteria.
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

        This method performs an N-step constrained beam search, which allows
        for a specified number of expansions (nstep) during the decoding
        process. The search is based on the approach outlined in the paper:
        "N-step Constrained Beam Search for Sequence-to-Sequence Models".
        For any usage outside ESPnet, please reference ESPnet (b-flo, PR #2444)
        until further modifications.
        Based on/Modified from https://arxiv.org/pdf/2002.03577.pdf.

        Args:
            enc_out: Encoder output sequence with shape (T, D_enc), where T is
                      the length of the sequence and D_enc is the dimension
                      of the encoder output.

        Returns:
            nbest_hyps: A list of N-best hypotheses sorted by their scores.

        Examples:
            >>> # Assuming enc_out is a tensor of shape (T, D_enc)
            >>> nbest_hyps = beam_search_transducer.nsc_beam_search(enc_out)
            >>> for hyp in nbest_hyps:
            >>>     print(f"Score: {hyp.score}, Sequence: {hyp.yseq}")

        Note:
            This implementation is designed to optimize the decoding process
            for transducer models, providing flexibility in the number of
            allowed expansions and the ability to incorporate language models
            if available.

        Raises:
            ValueError: If the shape of enc_out is not compatible or if nstep
                        is less than 1.
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
        Perform modified Adaptive Expansion Search (mAES) for decoding.

        This method implements the modified Adaptive Expansion Search 
        algorithm, which expands hypotheses based on the score and allows 
        for efficient search in the decoding process. It utilizes a 
        prefix search strategy combined with adaptive expansion to enhance 
        performance.

        The algorithm is based on and modified from:
        - https://ieeexplore.ieee.org/document/9250505
        - N-step constrained beam search (NSC).

        Args:
            enc_out: Tensor of shape (T, D_enc) representing the encoder 
                output sequence, where T is the sequence length and D_enc 
                is the encoder output dimension.

        Returns:
            List[ExtendedHypothesis]: A list of N-best hypotheses sorted 
            by their scores.

        Examples:
            >>> # Assuming `encoder_output` is a valid tensor from the encoder
            >>> search = BeamSearchTransducer(...)  # Initialize with parameters
            >>> best_hyps = search.modified_adaptive_expansion_search(encoder_output)
            >>> for hyp in best_hyps:
            >>>     print(hyp.yseq, hyp.score)

        Note:
            The number of expansions and scoring mechanisms can be adjusted 
            via the class parameters to optimize performance based on the 
            specific use case.

        Raises:
            NotImplementedError: If the search type or configuration is not 
            supported or implemented.
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

    def multi_blank_greedy_search(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """
        Greedy search for Multi-Blank Transducer (Multi-Blank Greedy, MBG).

    This implementation assumes:
    1. The index of the standard blank is the last entry of 
       `self.multi_blank_indices` rather than `self.blank_id` to 
       minimize changes to the original transducer.
    2. Other entries in `self.multi_blank_indices` represent large 
       blanks that account for multiple frames.

    The algorithm processes the encoder output and generates a 
    sequence of hypotheses by selectively predicting tokens based on 
    the state of the decoder and the encoder output.

    Args:
        enc_out: Encoder output sequence. Shape: (T, D_enc)

    Returns:
        List[Hypothesis]: A list containing the 1-best hypothesis.

    Examples:
        >>> enc_out = torch.randn(10, 256)  # Example encoder output
        >>> search = BeamSearchTransducer(...)  # Initialize with required params
        >>> best_hypothesis = search.multi_blank_greedy_search(enc_out)
        >>> print(best_hypothesis[0].yseq)  # Display the predicted sequence

    Note:
        This search method is particularly useful in scenarios where 
        the model needs to account for varying lengths of blank tokens 
        in the output sequence.
        """

        big_blank_duration = 1
        blank_start = self.multi_blank_indices[0]
        blank_end = self.multi_blank_indices[-1]

        dec_state = self.decoder.init_state(1)
        hyp = Hypothesis(score=0.0, yseq=[blank_end], dec_state=dec_state)
        cache = {}

        for enc_out_t in enc_out:
            # case 1: skip frames until big_blank_duration == 1
            if big_blank_duration > 1:
                big_blank_duration -= 1
                continue

            symbols_added = 0
            while symbols_added <= 3:
                dec_out, state, _ = self.decoder.score(hyp, cache)
                logp = torch.log_softmax(self.joint_network(enc_out_t, dec_out), dim=-1)
                top_logp, k = torch.max(logp, dim=-1)

                # case 2: predict a blank token
                if blank_start <= k <= blank_end:
                    big_blank_duration = self.multi_blank_durations[k - blank_start]
                    hyp.score += top_logp
                    break

                # case 3: predict a non-blank token
                else:
                    symbols_added += 1
                    hyp.yseq.append(int(k))
                    hyp.score += float(top_logp)
                    hyp.dec_state = state

        return [hyp]
