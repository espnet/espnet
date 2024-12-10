"""Search algorithms for Transducer models."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch

from espnet2.asr_transducer.decoder.abs_decoder import AbsDecoder
from espnet2.asr_transducer.joint_network import JointNetwork


@dataclass
class Hypothesis:
    """
    Search algorithms for Transducer models.

    This module implements search algorithms for Transducer models, including 
    the Beam Search algorithm. The `Hypothesis` class defines the default 
    hypothesis structure used in these search algorithms.

    Classes:
        Hypothesis: Represents a single hypothesis with its associated score, 
            label sequence, and states.
        ExtendedHypothesis: An extension of the Hypothesis class that includes 
            decoder output and language model scores.
        BeamSearchTransducer: Implements beam search for transducer models.

    Attributes:
        Hypothesis:
            score: Total log-probability of the hypothesis.
            yseq: Label sequence represented as a list of integer IDs.
            dec_state: RNN/MEGA Decoder state (None if stateless).
            lm_state: RNNLM state, can be a tuple of (N, D_lm) or None.
        ExtendedHypothesis:
            dec_out: Decoder output sequence of shape (B, D_dec).
            lm_score: Log-probabilities of the language model for given labels 
                of shape (vocab_size).
        BeamSearchTransducer:
            decoder: Decoder module used in the beam search.
            joint_network: Joint network module for scoring.
            beam_size: Size of the beam for search.
            lm: Language model module for scoring.
            lm_weight: Weight for the language model during scoring.
            search_type: Type of search algorithm to use.
            max_sym_exp: Maximum symbol expansions at each time step.
            u_max: Maximum expected target sequence length.
            nstep: Maximum expansion steps at each time step.
            expansion_gamma: Allowed log-probability difference for pruning.
            expansion_beta: Additional candidates for hypothesis selection.
            score_norm: Whether to normalize final scores by length.
            nbest: Number of final hypotheses to return.
            streaming: Whether to perform chunk-by-chunk beam search.

    Args:
        Hypothesis:
            score (float): Total log-probability of the hypothesis.
            yseq (List[int]): Sequence of label IDs.
            dec_state (Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]]): 
                Decoder state.
            lm_state (Optional[Union[Dict[str, Any], List[Any]]]): Language model state.
        ExtendedHypothesis:
            dec_out (torch.Tensor, optional): Decoder output sequence.
            lm_score (torch.Tensor, optional): Log-probabilities of LM.
        BeamSearchTransducer:
            decoder (AbsDecoder): Decoder module for the transducer.
            joint_network (JointNetwork): Joint network module for scoring.
            beam_size (int): Size of the beam.
            lm (Optional[torch.nn.Module]): Language model module.
            lm_weight (float): Weight for language model in scoring.
            search_type (str): Algorithm to use during inference.
            max_sym_exp (int): Maximum symbol expansions at each time step.
            u_max (int): Maximum expected target sequence length.
            nstep (int): Maximum expansion steps at each time step.
            expansion_gamma (float): Allowed log-probability difference for pruning.
            expansion_beta (int): Additional candidates for selection.
            score_norm (bool): Whether to normalize scores by length.
            nbest (int): Number of final hypotheses to return.
            streaming (bool): Whether to perform chunk-by-chunk beam search.

    Returns:
        Hypothesis: A list of n-best hypotheses after performing beam search.

    Examples:
        >>> hyp = Hypothesis(score=0.0, yseq=[1, 2, 3])
        >>> hyp.score
        0.0
        >>> hyp.yseq
        [1, 2, 3]

        >>> transducer = BeamSearchTransducer(decoder, joint_network, beam_size=5)
        >>> results = transducer(enc_out)
        >>> len(results)
        5

    Note:
        The `Hypothesis` class is designed to store and manage the state of 
        hypotheses during the beam search process.
    """

    score: float
    yseq: List[int]
    dec_state: Optional[Tuple[torch.Tensor, Optional[torch.Tensor]]] = None
    lm_state: Optional[Union[Dict[str, Any], List[Any]]] = None


@dataclass
class ExtendedHypothesis(Hypothesis):
    """
    Extended hypothesis definition for NSC beam search and mAES.

    This class extends the default hypothesis to include additional attributes
    specifically useful for the NSC beam search and modified Adaptive
    Expansion Search (mAES) algorithms.

    Attributes:
        dec_out: Decoder output sequence. Shape: (B, D_dec)
        lm_score: Log-probabilities of the language model for the given label.
                  Shape: (vocab_size)

    Args:
        score: Total log-probability.
        yseq: Label sequence as integer ID sequence.
        dec_state: RNN/MEGA Decoder state (None if Stateless).
        lm_state: RNNLM state. ((N, D_lm), (N, D_lm)) or None.
        dec_out: Decoder output sequence. Shape: (B, D_dec).
        lm_score: Log-probabilities of the LM for given label. Shape: (vocab_size).

    Examples:
        >>> hyp = ExtendedHypothesis(
        ...     score=-1.0,
        ...     yseq=[1, 2, 3],
        ...     dec_state=(torch.tensor([0.5]), None),
        ...     lm_state=None,
        ...     dec_out=torch.tensor([[0.1, 0.2], [0.3, 0.4]]),
        ...     lm_score=torch.tensor([0.5, 0.6])
        ... )
        >>> print(hyp.score)
        -1.0
        >>> print(hyp.yseq)
        [1, 2, 3]
    """

    dec_out: torch.Tensor = None
    lm_score: torch.Tensor = None


class BeamSearchTransducer:
    """
    Beam search implementation for Transducer models.

    This class implements a beam search algorithm for transducer models in 
    automatic speech recognition (ASR). It is designed to work with a decoder 
    and a joint network module to produce N-best hypotheses from the encoder 
    output.

    Attributes:
        decoder (AbsDecoder): Decoder module for generating sequences.
        joint_network (JointNetwork): Joint network module that combines 
            encoder and decoder outputs.
        beam_size (int): Size of the beam for search.
        lm (Optional[torch.nn.Module]): Language model module for soft fusion.
        lm_weight (float): Weight for the language model in scoring.
        search_type (str): Type of search algorithm used during inference.
        max_sym_exp (int): Maximum symbol expansions at each time step.
        u_max (int): Maximum expected target sequence length.
        nstep (int): Maximum expansion steps at each time step.
        expansion_gamma (float): Log probability difference for pruning.
        expansion_beta (int): Additional candidates for expanded hypotheses.
        score_norm (bool): Whether to normalize final scores by length.
        nbest (int): Number of final hypotheses.
        streaming (bool): Whether to perform chunk-by-chunk beam search.

    Args:
        decoder (AbsDecoder): The decoder module.
        joint_network (JointNetwork): The joint network module.
        beam_size (int): The size of the beam for search.
        lm (Optional[torch.nn.Module]): The language model for soft fusion.
        lm_weight (float): Weight of the language model.
        search_type (str): Type of search algorithm to use.
        max_sym_exp (int): Maximum symbol expansions.
        u_max (int): Maximum expected target sequence length.
        nstep (int): Maximum number of expansion steps.
        expansion_gamma (float): Log probability difference for pruning.
        expansion_beta (int): Additional candidates for expanded hypotheses.
        score_norm (bool): Normalize final scores by length.
        nbest (int): Number of final hypotheses.
        streaming (bool): Perform chunk-by-chunk beam search.

    Examples:
        >>> beam_search = BeamSearchTransducer(decoder, joint_network, beam_size=5)
        >>> hypotheses = beam_search(enc_out, is_final=True)

    Raises:
        NotImplementedError: If the specified search type is not supported.

    Note:
        Ensure that the `beam_size` is less than or equal to the vocabulary size 
        of the decoder.
    """

    def __init__(
        self,
        decoder: AbsDecoder,
        joint_network: JointNetwork,
        beam_size: int,
        lm: Optional[torch.nn.Module] = None,
        lm_weight: float = 0.1,
        search_type: str = "default",
        max_sym_exp: int = 3,
        u_max: int = 50,
        nstep: int = 2,
        expansion_gamma: float = 2.3,
        expansion_beta: int = 2,
        score_norm: bool = False,
        nbest: int = 1,
        streaming: bool = False,
    ) -> None:
        """Construct a BeamSearchTransducer object."""
        super().__init__()

        self.decoder = decoder
        self.joint_network = joint_network

        self.vocab_size = decoder.vocab_size

        assert beam_size <= self.vocab_size, (
            "beam_size (%d) should be smaller than or equal to vocabulary size (%d)."
            % (
                beam_size,
                self.vocab_size,
            )
        )
        self.beam_size = beam_size

        if search_type == "default":
            self.search_algorithm = self.default_beam_search
        elif search_type == "tsd":
            assert max_sym_exp > 1, "max_sym_exp (%d) should be greater than one." % (
                max_sym_exp
            )
            self.max_sym_exp = max_sym_exp

            self.search_algorithm = self.time_sync_decoding
        elif search_type == "alsd":
            assert not streaming, "ALSD is not available in streaming mode."

            assert u_max >= 0, "u_max should be a positive integer, a portion of max_T."
            self.u_max = u_max

            self.search_algorithm = self.align_length_sync_decoding
        elif search_type == "maes":
            assert self.vocab_size >= beam_size + expansion_beta, (
                "beam_size (%d) + expansion_beta (%d) "
                " should be smaller than or equal to vocab size (%d)."
                % (beam_size, expansion_beta, self.vocab_size)
            )
            self.max_candidates = beam_size + expansion_beta

            self.nstep = nstep
            self.expansion_gamma = expansion_gamma

            self.search_algorithm = self.modified_adaptive_expansion_search
        else:
            raise NotImplementedError(
                "Specified search type (%s) is not supported." % search_type
            )

        self.use_lm = lm is not None

        if self.use_lm:
            assert hasattr(lm, "rnn_type"), "Transformer LM is currently not supported."

            self.sos = self.vocab_size - 1

            self.lm = lm
            self.lm_weight = lm_weight

        self.score_norm = score_norm
        self.nbest = nbest

        self.reset_cache()

    def __call__(
        self,
        enc_out: torch.Tensor,
        is_final: bool = True,
    ) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)
            is_final: Whether enc_out is the final chunk of data.

        Returns:
            nbest_hyps: N-best decoding results

        """
        self.decoder.set_device(enc_out.device)

        hyps = self.search_algorithm(enc_out)

        if is_final:
            self.reset_cache()

            return self.sort_nbest(hyps)

        self.search_cache = hyps

        return hyps

    def reset_cache(self) -> None:
        """
        Reset cache for streaming decoding.

        This method clears the score cache in the decoder and resets the search
        cache. It is particularly useful in scenarios where multiple decoding
        chunks are processed in a streaming manner, ensuring that previous
        state information does not interfere with subsequent decoding steps.

        Attributes:
            score_cache (dict): A dictionary used by the decoder to cache scores.
            search_cache (None): A placeholder for caching hypotheses during
                the search process.

        Examples:
            >>> beam_search = BeamSearchTransducer(...)
            >>> beam_search.reset_cache()  # Resets caches before new decoding.

        Note:
            This method is automatically called at the end of a decoding
            pass if `is_final` is set to `True` in the `__call__` method.
        """
        self.decoder.score_cache = {}
        self.search_cache = None

    def sort_nbest(self, hyps: List[Hypothesis]) -> List[Hypothesis]:
        """
        Sort in-place hypotheses by score or score given sequence length.

        This method sorts a list of hypotheses based on their scores. If 
        `score_norm` is set to `True`, it normalizes the scores by the 
        length of the corresponding label sequences. The sorted list 
        will contain only the top `nbest` hypotheses.

        Args:
            hyps: A list of `Hypothesis` instances to be sorted.

        Returns:
            List[Hypothesis]: A sorted list of hypotheses, containing only 
            the top `nbest` hypotheses based on their scores.

        Examples:
            >>> hyp1 = Hypothesis(score=10.0, yseq=[1, 2])
            >>> hyp2 = Hypothesis(score=15.0, yseq=[1, 3])
            >>> hyp3 = Hypothesis(score=5.0, yseq=[2, 3])
            >>> sorted_hyps = sort_nbest([hyp1, hyp2, hyp3])
            >>> sorted_hyps[0].score  # Should return 15.0
            15.0

        Note:
            The sorting is done in-place, meaning the original list 
            `hyps` will be modified.
        """
        if self.score_norm:
            hyps.sort(key=lambda x: x.score / len(x.yseq), reverse=True)
        else:
            hyps.sort(key=lambda x: x.score, reverse=True)

        return hyps[: self.nbest]

    def recombine_hyps(self, hyps: List[Hypothesis]) -> List[Hypothesis]:
        """
        Recombine hypotheses with same label ID sequence.

        This method aggregates the scores of hypotheses that share the same
        label ID sequence, effectively merging them into a single hypothesis
        with a combined score. The score is computed using the log-sum-exp
        trick to prevent numerical underflow.

        Args:
            hyps: A list of Hypothesis objects that need to be recombined.

        Returns:
            final: A list of recombined Hypothesis objects, where each unique
            label ID sequence is represented by a single Hypothesis with its
            score adjusted accordingly.

        Examples:
            >>> hyps = [
            ...     Hypothesis(score=1.0, yseq=[1, 2]),
            ...     Hypothesis(score=0.5, yseq=[1, 2]),
            ...     Hypothesis(score=2.0, yseq=[3, 4]),
            ... ]
            >>> recombined = recombine_hyps(hyps)
            >>> for hyp in recombined:
            ...     print(hyp.yseq, hyp.score)
            [1, 2] -1.2039728043259318  # logaddexp(1.0, 0.5)
            [3, 4] 2.0
        """
        final = {}

        for hyp in hyps:
            str_yseq = "_".join(map(str, hyp.yseq))

            if str_yseq in final:
                final[str_yseq].score = np.logaddexp(final[str_yseq].score, hyp.score)
            else:
                final[str_yseq] = hyp

        return [*final.values()]

    def select_k_expansions(
        self,
        hyps: List[ExtendedHypothesis],
        topk_idx: torch.Tensor,
        topk_logp: torch.Tensor,
    ) -> List[ExtendedHypothesis]:
        """
        Return K hypotheses candidates for expansion from a list of hypotheses.

        K candidates are selected according to the extended hypotheses probabilities
        and a prune-by-value method. Where K is equal to beam_size + beta.

        Args:
            hyps: List of extended hypotheses to select from.
            topk_idx: Indices of candidate hypotheses.
            topk_logp: Log-probabilities of candidate hypotheses.

        Returns:
            k_expansions: List of the best K expansion hypotheses candidates.

        Examples:
            >>> hyps = [ExtendedHypothesis(yseq=[0], score=1.0),
            ...          ExtendedHypothesis(yseq=[1], score=0.8)]
            >>> topk_idx = torch.tensor([[0, 1], [0, 1]])
            >>> topk_logp = torch.tensor([[0.5, 0.3], [0.4, 0.2]])
            >>> k_expansions = select_k_expansions(hyps, topk_idx, topk_logp)
            >>> print(k_expansions)
            [<ExtendedHypothesis>, <ExtendedHypothesis>]
        """
        k_expansions = []

        for i, hyp in enumerate(hyps):
            hyp_i = [
                (int(k), hyp.score + float(v))
                for k, v in zip(topk_idx[i], topk_logp[i])
            ]
            k_best_exp = max(hyp_i, key=lambda x: x[1])[1]

            k_expansions.append(
                sorted(
                    filter(
                        lambda x: (k_best_exp - self.expansion_gamma) <= x[1], hyp_i
                    ),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )

        return k_expansions

    def create_lm_batch_inputs(self, hyps_seq: List[List[int]]) -> torch.Tensor:
        """
        Make batch of inputs with left padding for LM scoring.

        This function creates a padded batch of hypothesis sequences, where each
        sequence is left-padded with a start-of-sequence token and right-padded
        with zeros to ensure that all sequences in the batch have the same length.

        Args:
            hyps_seq: A list of hypothesis sequences, where each sequence is a list
                of integers representing label IDs.

        Returns:
            torch.Tensor: A tensor containing the padded batch of sequences. The
                shape of the tensor will be (batch_size, max_length), where
                max_length is the length of the longest sequence in the input.

        Examples:
            >>> hyps_seq = [[1, 2, 3], [4, 5], [6]]
            >>> batch_inputs = create_lm_batch_inputs(hyps_seq)
            >>> print(batch_inputs)
            tensor([[ 0,  1,  2,  3],
                    [ 0,  4,  5,  0],
                    [ 0,  6,  0,  0]])

        Note:
            The start-of-sequence token is defined as `self.sos`, and zero is used
            for padding.
        """
        max_len = max([len(h) for h in hyps_seq])

        return torch.LongTensor(
            [[self.sos] + ([0] * (max_len - len(h))) + h[1:] for h in hyps_seq],
            device=self.decoder.device,
        )

    def default_beam_search(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """
        Beam search implementation without prefix search.

        This method performs a beam search over the output of the encoder 
        without using prefix search. It evaluates the hypotheses at each 
        time step, expanding the most promising ones according to the beam 
        size and the scores computed from the joint network.

        Modified from: https://arxiv.org/pdf/1211.3711.pdf

        Args:
            enc_out: Encoder output sequence. Shape (T, D).

        Returns:
            nbest_hyps: List of N-best hypotheses sorted by their scores.

        Examples:
            >>> enc_out = torch.randn(10, 256)  # Example encoder output
            >>> beam_search = BeamSearchTransducer(decoder, joint_network, beam_size=5)
            >>> results = beam_search.default_beam_search(enc_out)
            >>> print(results)  # List of Hypothesis objects with their scores and sequences.

        Note:
            The hypotheses are scored based on both the decoder output and 
            the language model (if available), and are pruned according 
            to the beam size at each time step.
        """
        beam_k = min(self.beam_size, (self.vocab_size - 1))
        max_t = len(enc_out)

        if self.search_cache is not None:
            kept_hyps = self.search_cache
        else:
            kept_hyps = [
                Hypothesis(
                    score=0.0,
                    yseq=[0],
                    dec_state=self.decoder.init_state(1),
                )
            ]

        for t in range(max_t):
            hyps = kept_hyps
            kept_hyps = []

            while True:
                max_hyp = max(hyps, key=lambda x: x.score)
                hyps.remove(max_hyp)

                dec_out, state = self.decoder.score(
                    max_hyp.yseq,
                    max_hyp.dec_state,
                )

                logp = torch.log_softmax(
                    self.joint_network(enc_out[t : t + 1, :], dec_out),
                    dim=-1,
                ).squeeze(0)
                top_k = logp[1:].topk(beam_k, dim=-1)

                kept_hyps.append(
                    Hypothesis(
                        score=(max_hyp.score + float(logp[0:1])),
                        yseq=max_hyp.yseq,
                        dec_state=max_hyp.dec_state,
                        lm_state=max_hyp.lm_state,
                    )
                )

                if self.use_lm:
                    lm_scores, lm_state = self.lm.score(
                        torch.LongTensor(
                            [self.sos] + max_hyp.yseq[1:], device=self.decoder.device
                        ),
                        max_hyp.lm_state,
                        None,
                    )
                else:
                    lm_state = max_hyp.lm_state

                for logp, k in zip(*top_k):
                    score = max_hyp.score + float(logp)

                    if self.use_lm:
                        score += self.lm_weight * lm_scores[k + 1]

                    hyps.append(
                        Hypothesis(
                            score=score,
                            yseq=max_hyp.yseq + [int(k + 1)],
                            dec_state=state,
                            lm_state=lm_state,
                        )
                    )

                hyps_max = float(max(hyps, key=lambda x: x.score).score)
                kept_most_prob = sorted(
                    [hyp for hyp in kept_hyps if hyp.score > hyps_max],
                    key=lambda x: x.score,
                )

                if len(kept_most_prob) >= self.beam_size:
                    kept_hyps = kept_most_prob
                    break

        return kept_hyps

    def align_length_sync_decoding(
        self,
        enc_out: torch.Tensor,
    ) -> List[Hypothesis]:
        """
        Alignment-length synchronous beam search implementation.

        This method performs a beam search that synchronizes the length of the 
        generated sequences with the input encoder outputs. The search is 
        based on the algorithm described in the paper:
        "A Generalized Beam Search Algorithm for Sequence-to-Sequence 
        Learning" (https://ieeexplore.ieee.org/document/9053040).

        Args:
            enc_out: Encoder output sequences. Shape is (T, D) where T is 
                    the number of time steps and D is the dimension of 
                    the encoder output.

        Returns:
            List[Hypothesis]: A list of N-best hypotheses generated from 
                            the beam search.

        Examples:
            >>> beam_search_transducer = BeamSearchTransducer(...)
            >>> encoder_output = torch.randn(10, 256)  # Example encoder output
            >>> hypotheses = beam_search_transducer.align_length_sync_decoding(encoder_output)
            >>> for hyp in hypotheses:
            ...     print(hyp.yseq, hyp.score)
        """
        t_max = int(enc_out.size(0))
        u_max = min(self.u_max, (t_max - 1))

        B = [Hypothesis(yseq=[0], score=0.0, dec_state=self.decoder.init_state(1))]
        final = []

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
                beam_enc_out = torch.stack([b[1] for b in B_enc_out])
                beam_dec_out, beam_state = self.decoder.batch_score(B_)

                beam_logp = torch.log_softmax(
                    self.joint_network(beam_enc_out, beam_dec_out),
                    dim=-1,
                )
                beam_topk = beam_logp[:, 1:].topk(self.beam_size, dim=-1)

                if self.use_lm:
                    beam_lm_scores, beam_lm_states = self.lm.batch_score(
                        self.create_lm_batch_inputs([b.yseq for b in B_]),
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

                B = sorted(A, key=lambda x: x.score, reverse=True)[: self.beam_size]
                B = self.recombine_hyps(B)

        if final:
            return final

        return B

    def time_sync_decoding(self, enc_out: torch.Tensor) -> List[Hypothesis]:
        """
        Time synchronous beam search implementation.

        This method implements a beam search algorithm that operates in a time 
        synchronous manner. It takes the encoder output sequence and generates 
        N-best hypotheses based on the joint network's log-probabilities. The 
        approach allows for multiple symbol expansions at each time step, making 
        it suitable for scenarios where temporal alignment is critical.

        Args:
            enc_out: Encoder output sequence. (T, D)

        Returns:
            List[Hypothesis]: N-best hypotheses, sorted by their scores.

        Examples:
            >>> decoder = AbsDecoder(...)
            >>> joint_network = JointNetwork(...)
            >>> beam_search = BeamSearchTransducer(decoder, joint_network, beam_size=5)
            >>> enc_out = torch.randn(10, decoder.input_dim)  # Example encoder output
            >>> nbest_hyps = beam_search.time_sync_decoding(enc_out)
            >>> for hyp in nbest_hyps:
            >>>     print(hyp.yseq, hyp.score)

        Note:
            The method can utilize a language model if one is provided during the 
            initialization of the `BeamSearchTransducer`.

        Raises:
            RuntimeError: If the input tensor dimensions do not match expected shapes.
        """
        if self.search_cache is not None:
            B = self.search_cache
        else:
            B = [
                Hypothesis(
                    yseq=[0],
                    score=0.0,
                    dec_state=self.decoder.init_state(1),
                )
            ]

            if self.use_lm:
                B[0].lm_state = self.lm.zero_state()

        for enc_out_t in enc_out:
            A = []
            C = B

            enc_out_t = enc_out_t.unsqueeze(0)

            for v in range(self.max_sym_exp):
                D = []

                beam_dec_out, beam_state = self.decoder.batch_score(C)

                beam_logp = torch.log_softmax(
                    self.joint_network(enc_out_t, beam_dec_out),
                    dim=-1,
                )
                beam_topk = beam_logp[:, 1:].topk(self.beam_size, dim=-1)

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
                            self.create_lm_batch_inputs([c.yseq for c in C]),
                            [c.lm_state for c in C],
                            None,
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

                C = sorted(D, key=lambda x: x.score, reverse=True)[: self.beam_size]

            B = sorted(A, key=lambda x: x.score, reverse=True)[: self.beam_size]

        return B

    def modified_adaptive_expansion_search(
        self,
        enc_out: torch.Tensor,
    ) -> List[ExtendedHypothesis]:
        """
        Modified version of Adaptive Expansion Search (mAES).

        This method implements a modified version of the Adaptive Expansion
        Search algorithm for beam search decoding in transducer models. It 
        utilizes a combination of hypotheses from previous steps and expands 
        them based on the current encoder output.

        Based on the original Adaptive Expansion Search (AES) as described in
        https://ieeexplore.ieee.org/document/9250505 and the Non-Stationary
        Context (NSC) approach from https://arxiv.org/abs/2201.05420.

        Args:
            enc_out: Encoder output sequence. (T, D_enc)

        Returns:
            nbest_hyps: N-best hypotheses sorted by score.

        Examples:
            >>> enc_out = torch.rand(10, 256)  # Example encoder output
            >>> beam_search = BeamSearchTransducer(...)
            >>> nbest_hyps = beam_search.modified_adaptive_expansion_search(enc_out)
        """
        if self.search_cache is not None:
            kept_hyps = self.search_cache
        else:
            init_tokens = [
                ExtendedHypothesis(
                    yseq=[0],
                    score=0.0,
                    dec_state=self.decoder.init_state(1),
                )
            ]

            beam_dec_out, beam_state = self.decoder.batch_score(
                init_tokens,
            )

            if self.use_lm:
                beam_lm_scores, beam_lm_states = self.lm.batch_score(
                    self.create_lm_batch_inputs([h.yseq for h in init_tokens]),
                    [h.lm_state for h in init_tokens],
                    None,
                )

                lm_state = beam_lm_states[0]
                lm_score = beam_lm_scores[0]
            else:
                lm_state = None
                lm_score = None

            kept_hyps = [
                ExtendedHypothesis(
                    yseq=[0],
                    score=0.0,
                    dec_state=self.decoder.select_state(beam_state, 0),
                    dec_out=beam_dec_out[0],
                    lm_state=lm_state,
                    lm_score=lm_score,
                )
            ]

        for enc_out_t in enc_out:
            hyps = kept_hyps
            kept_hyps = []

            beam_enc_out = enc_out_t.unsqueeze(0)

            list_b = []
            for n in range(self.nstep):
                beam_dec_out = torch.stack([h.dec_out for h in hyps])

                beam_logp, beam_idx = torch.log_softmax(
                    self.joint_network(beam_enc_out, beam_dec_out),
                    dim=-1,
                ).topk(self.max_candidates, dim=-1)

                k_expansions = self.select_k_expansions(hyps, beam_idx, beam_logp)

                list_exp = []
                for i, hyp in enumerate(hyps):
                    for k, new_score in k_expansions[i]:
                        new_hyp = ExtendedHypothesis(
                            yseq=hyp.yseq[:],
                            score=new_score,
                            dec_out=hyp.dec_out,
                            dec_state=hyp.dec_state,
                            lm_state=hyp.lm_state,
                            lm_score=hyp.lm_score,
                        )

                        if k == 0:
                            list_b.append(new_hyp)
                        else:
                            new_hyp.yseq.append(int(k))

                            if self.use_lm:
                                new_hyp.score += self.lm_weight * float(hyp.lm_score[k])

                            list_exp.append(new_hyp)

                if not list_exp:
                    kept_hyps = sorted(
                        self.recombine_hyps(list_b), key=lambda x: x.score, reverse=True
                    )[: self.beam_size]

                    break
                else:
                    beam_dec_out, beam_state = self.decoder.batch_score(
                        list_exp,
                    )

                    if self.use_lm:
                        beam_lm_scores, beam_lm_states = self.lm.batch_score(
                            self.create_lm_batch_inputs([h.yseq for h in list_exp]),
                            [h.lm_state for h in list_exp],
                            None,
                        )

                    if n < (self.nstep - 1):
                        for i, hyp in enumerate(list_exp):
                            hyp.dec_out = beam_dec_out[i]
                            hyp.dec_state = self.decoder.select_state(beam_state, i)

                            if self.use_lm:
                                hyp.lm_state = beam_lm_states[i]
                                hyp.lm_score = beam_lm_scores[i]

                        hyps = list_exp[:]
                    else:
                        beam_logp = torch.log_softmax(
                            self.joint_network(beam_enc_out, beam_dec_out),
                            dim=-1,
                        )

                        for i, hyp in enumerate(list_exp):
                            hyp.score += float(beam_logp[i, 0])

                            hyp.dec_out = beam_dec_out[i]
                            hyp.dec_state = self.decoder.select_state(beam_state, i)

                            if self.use_lm:
                                hyp.lm_state = beam_lm_states[i]
                                hyp.lm_score = beam_lm_scores[i]

                        kept_hyps = sorted(
                            self.recombine_hyps(list_b + list_exp),
                            key=lambda x: x.score,
                            reverse=True,
                        )[: self.beam_size]

        return kept_hyps
