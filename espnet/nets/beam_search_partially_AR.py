# Beam search module for partially autoregressive decoding.
# Copyright 2024 Masao Someki
# This script is licensed under MIT license.
# This script is the upgraded version used in https://arxiv.org/abs/2309.14922
import logging
from typing import Any, Dict, List, NamedTuple, Tuple, Union

import torch
from packaging.version import parse as V

from espnet.nets.batch_beam_search import BatchBeamSearch, BatchHypothesis
from espnet.nets.e2e_asr_common import end_detect

is_torch_1_9_plus = V(torch.__version__) >= V("1.9.0")


class Hypothesis(NamedTuple):
    """Hypothesis data type."""

    yseq: torch.Tensor
    score: Union[float, torch.Tensor] = 0
    scores: Dict[str, Union[float, torch.Tensor]] = dict()
    states: Dict[str, Any] = dict()

    def asdict(self) -> dict:
        """Convert data to JSON-friendly dict."""
        return self._replace(
            yseq=self.yseq.tolist(),
            score=float(self.score),
        )._asdict()


class PartiallyARHypothesis(NamedTuple):
    """Hypothesis data type for partially autoregressive decoding."""

    yseq: torch.Tensor
    score: Union[float, torch.Tensor] = None
    states: Dict[str, Any] = dict()
    yseq_length: torch.Tensor = torch.tensor([])
    eos: torch.Tensor = torch.tensor([])

    def asdict(self) -> dict:
        """Convert data to JSON-friendly dict."""
        return self._replace(
            yseq=self.yseq.tolist(),
            score=float(self.score),
        )._asdict()


class PartiallyARBeamSearch(BatchBeamSearch):
    """Partially autoregressive beam search implementation.
    Partially autoregressive hypothesis is a set of BatchHypothesis.

    We need to use `add_mask` function to add a hypothesis for a mask.
    Before search and beam search method, each partially autoregressive
    hypothesis is extracted to BatchHypothesis, and applied the same process
    as the batched_beam_search.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.masks = []
        self.mask_ids = None
        self.beam_ids = None
        self.num_hyps_for_masks = None
        self.beam_arange = torch.arange(self.beam_size).unsqueeze(1)
        self.n_vocab = self.n_vocab - 1  # remove mask token

    def init_masks(self):
        self.masks = []
        self.num_hyps_for_masks = []

    def add_mask(self, primer: List[int], eos: int):
        """Add a mask to a batch of hypotheses.

        Args:
            primer (torch.Tensor): Primer yseq.

        """
        self.masks.append((primer, eos))

    def init_hyp(self, x: torch.Tensor) -> PartiallyARHypothesis:
        """Get an initial hypothesis data for each mask.

        Args:
            x (torch.Tensor): The encoder output feature

        Returns:
            PartiallyARHypothesis: The initial hypothesis.

        """
        assert len(self.masks) > 0, "add_mask must be called before init_hyp"

        init_states = dict()  # Dict[str, List[torch.Tensor]]
        for k, d in self.scorers.items():
            init_state = d.batch_init_state(x)
            init_states[k] = [init_state for _ in range(len(self.masks))]

        longest_yseq = max([len(m[0]) for m in self.masks])
        yseq_tensor = torch.zeros(
            (len(self.masks), self.beam_size, longest_yseq),
            dtype=torch.long,
            device=x.device,
        )  # (n_mask, n_beam, max_yseq)
        yseq_length = torch.zeros((len(self.masks)), dtype=torch.long, device=x.device)
        score = torch.zeros(
            (len(self.masks) * self.beam_size), dtype=x.dtype, device=x.device
        )
        eoses = torch.zeros(len(self.masks), dtype=torch.long, device=x.device)
        self.num_hyps_for_masks = torch.ones(
            len(self.masks), dtype=torch.long, device=x.device
        )
        self.beam_arange = self.beam_arange.to(x.device)

        for i, m in enumerate(self.masks):
            yseq_tensor[i, 0, : len(m[0])] = torch.LongTensor(m[0])
            yseq_length[i] = len(m[0])
            eoses[i] = torch.LongTensor([m[1]])
            self.num_hyps_for_masks[i] = 1

        hyp = PartiallyARHypothesis(
            yseq=yseq_tensor,
            score=score,
            states=init_states,
            yseq_length=yseq_length,
            eos=eoses,
        )
        return hyp

    def score_full(
        self, hyp: PartiallyARHypothesis, x: torch.Tensor, is_first: bool = False
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Score new hypothesis by `self.full_scorers`.

        Args:
            hyp (PartiallyARHypothesis): Hypothesis with prefix tokens to score
            x (torch.Tensor): Corresponding input feature

        Returns:
            Tuple[Dict[str, torch.Tensor], Dict[str, Any]]: Tuple of
                score dict of `hyp` that has string keys of `self.full_scorers`
                and tensor score values of shape: `(self.n_vocab,)`,
                and state dict that has string keys
                and state values of `self.full_scorers`

        """
        new_scores = dict()
        new_states = dict()
        n_mask = len(self.masks)
        # Create batch of scores and states
        # If states[key] is None, then it is the first iteration,
        # and we cannot execute parallel computation over masks.
        # If states[key] is not None, then it is second or later iteration.
        # If decoder or lm is Transformer based model, we can apply parallel
        # computation beacause we only need the last sequence to compute scores.
        # Otherwise, we cannot apply parallel computation.`
        for k, d in self.full_scorers.items():
            mask_scores, mask_states = d.batch_score_partially_AR(
                hyp.yseq.reshape(
                    -1, hyp.yseq.shape[-1]
                ),  # (n_mask * n_beam, y_seq_len)
                hyp.states[k],
                x.expand(
                    hyp.yseq.size(0) * hyp.yseq.size(1), *x.shape
                ),  # (n_beam, *x.shape)
                hyp.yseq_length.unsqueeze(1)
                .expand(n_mask, self.beam_size)
                .reshape(-1),  # (n_mask * n_beam)
            )
            new_scores[k] = mask_scores
            new_states[k] = mask_states

        return new_scores, new_states

    def _select(
        self, hyps: PartiallyARHypothesis, i_mask: int, i_beam: int
    ) -> Hypothesis:
        return Hypothesis(
            yseq=hyps.yseq[i_mask, i_beam, : hyps.yseq_length[i_mask]],
            score=hyps.score.view(len(self.masks), -1)[i_mask, i_beam],
            scores=None,
            states=None,
        )

    def forward(self, x: torch.Tensor, max_seq_len: int = None) -> List[Hypothesis]:
        """Perform beam search.

        Args:
            x (torch.Tensor): Encoded speech feature (T, D)
            maxlenratio (float): Input length ratio to obtain max output length.
                If maxlenratio=0.0 (default), it uses a end-detect function
                to automatically find maximum hypothesis lengths
                If maxlenratio<0.0, its absolute value is interpreted
                as a constant max output length.
            minlenratio (float): Input length ratio to obtain min output length.

        Returns:
            list[Hypothesis]: N-best decoding results

        """
        # set length bounds
        if max_seq_len is not None:
            maxlen = max_seq_len
        else:
            maxlen = x.size(0)

        logging.info("decoder input length: " + str(x.shape[0]))
        logging.info("max output length: " + str(maxlen))

        # initialize mask ids
        self.mask_ids = torch.arange(len(self.masks)).view(-1, 1).to(x.device)

        # running_hyps is PartiallyARHypothesis.
        running_hyps = self.init_hyp(x)

        # ended_hyps will be list of ended_hyps for each masks
        ended_hyps = [[] for _ in range(len(self.masks))]

        for i in range(maxlen):
            logging.debug("position " + str(i))
            best = self.search(running_hyps, x)  # PartiallyARHypothesis

            # post process of one iteration
            # running_hyps is BatchHypothesis
            running_hyps = self.post_process(i, maxlen, best, ended_hyps)

            # end detection
            if end_detect([h.asdict() for eh in ended_hyps for h in eh], i):
                logging.info(f"end detected at {i}")
                break
            if len(running_hyps) == 0:
                logging.info("no hypothesis. Finish decoding.")
                break
            else:
                logging.debug(f"remained hypotheses: {len(running_hyps)}")

        nbest_hyps_for_masks = [
            sorted(ended_hyp, key=lambda x: x.score, reverse=True)
            for ended_hyp in ended_hyps
        ]
        # check the number of hypotheses reaching to eos
        for i, nbest_mask in enumerate(nbest_hyps_for_masks):
            if len(nbest_mask) == 0:
                logging.warning(f"there is no N-best results for mask {i}")

        bests = [nbest_mask[0] for nbest_mask in nbest_hyps_for_masks]
        self._log_bests(bests, maxlen, nbest_hyps_for_masks)
        return bests

    def batch_beam(
        self, weighted_scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Batch-compute topk full token ids and partial token ids.

        Args:
            weighted_scores (torch.Tensor): The weighted sum scores for each tokens.
                Its shape is `(n_beam, self.vocab_size)`.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                The topk full (prev_hyp, new_token) ids
                and partial (prev_hyp, new_token) ids.
                Their shapes are all `(self.beam_size,)`

        """
        weighted_scores = weighted_scores.view(len(self.masks), -1)
        top_ids = weighted_scores.topk(self.beam_size, dim=1)[1]  # (n_mask, beam_size)
        # Because of the flatten above, `top_ids` is organized as:
        # [hyp1 * V + token1, hyp2 * V + token2, ..., hypK * V + tokenK],
        # where V is `self.n_vocab` and K is `self.beam_size`
        if is_torch_1_9_plus:
            prev_hyp_ids = torch.div(top_ids, self.n_vocab, rounding_mode="trunc")
        else:
            prev_hyp_ids = top_ids // self.n_vocab
        new_token_ids = top_ids % self.n_vocab

        return prev_hyp_ids, new_token_ids  # (n_mask, n_beam)

    def search(
        self, running_hyps: PartiallyARHypothesis, x: torch.Tensor
    ) -> PartiallyARHypothesis:
        """Search new tokens for running hypotheses and encoded speech x.

        Args:
            running_hyps (BatchHypothesis): Running hypotheses on beam
            x (torch.Tensor): Encoded speech feature (T, D)

        Returns:
            BatchHypothesis: Best sorted hypotheses

        """
        weighted_scores = torch.zeros(
            len(self.masks) * self.beam_size,
            self.n_vocab,
            dtype=x.dtype,
            device=x.device,
        )  # (n_mask * n_beam, n_vocab)
        beam_mask = (
            (self.beam_arange >= self.num_hyps_for_masks).transpose(0, 1).reshape(-1, 1)
        )  # (n_beam, n_mask) -> (n_mask, n_beam) -> (n_mask * n_beam, 1)
        beam_mask = beam_mask * -100000.0

        # COMPUTE FULL SCORERS
        scores, states = self.score_full(
            running_hyps, x
        )  # scorers: (n_mask * n_beam, n_vocab)
        for k in self.full_scorers:
            weighted_scores += self.weights[k] * scores[k]

        # add previous hyp scores
        weighted_scores += running_hyps.score.unsqueeze(
            1
        )  # (n_mask * n_beam, n_vocab) + (n_mask * n_beam, 1)
        weighted_scores = (
            weighted_scores + beam_mask
        )  # (n_mask * n_beam, n_vocab). no score for padded hypos

        # COMPUTE BATCHED BEAM SEARCH
        # prev_hyp_ids and new_token_ids: (n_mask, beam_size)
        prev_hyp_ids, new_token_ids = self.batch_beam(weighted_scores)

        new_hyp = self._get_new_mask_parallel_hyp(
            running_hyps, prev_hyp_ids, new_token_ids, weighted_scores, states
        )

        return new_hyp

    def post_process(
        self,
        i: int,
        maxlen: int,
        running_hyps: PartiallyARHypothesis,
        ended_hyps: List[List[Hypothesis]],
    ) -> BatchHypothesis:
        """Perform post-processing of beam search iterations.
        Extract BatchHypothesis for each mask, and perform post-process.
        Then merge BatchHypothesis.

        Args:
            i (int): The length of hypothesis tokens.
            maxlen (int): The maximum length of tokens in beam search.
            maxlenratio (int): The maximum length ratio in beam search.
            running_hyps (BatchHypothesis): The running hypotheses in beam search.
            ended_hyps (List[Hypothesis]): The ended hypotheses in beam search.

        Returns:
            BatchHypothesis: The new running hypotheses.

        """
        n_mask = len(self.masks)
        # add eos in the final loop to avoid that there are no ended hyps
        if i == maxlen - 1 and any(
            [len(ended_hyps[i_mask]) == 0 for i_mask in range(n_mask)]
        ):
            yseq_shape = running_hyps.yseq.shape
            yseq_eos = torch.zeros(
                (n_mask, self.beam_size, yseq_shape[-1] + 1),
                dtype=torch.int64,
                device=running_hyps.yseq.device,
            )
            yseq_eos[:, :, :-1] = running_hyps.yseq
            yseq_eos[self.mask_ids, :, running_hyps.yseq_length.view(-1, 1)] = (
                running_hyps.eos.view(-1, 1, 1)
            )
            running_hyps = PartiallyARHypothesis(
                yseq=yseq_eos,
                score=running_hyps.score,
                yseq_length=running_hyps.yseq_length + 1,
                states=running_hyps.states,
                eos=running_hyps.eos,
            )

        # add ended hypotheses to a final list, and removed them from current hypotheses
        # (this will be a probmlem, number of hyps < beam)
        if self.beam_ids is None:
            self.beam_ids = torch.arange(
                self.beam_size, dtype=torch.int64, device=running_hyps.yseq.device
            )

        remained_ids_list = []
        for i_mask in range(len(self.masks)):
            # check if there is any ended hypo for mask i.
            is_eos = (
                running_hyps.yseq[
                    i_mask, self.beam_ids, running_hyps.yseq_length[i_mask] - 1
                ]
                == running_hyps.eos[i_mask]
            )
            for b in torch.nonzero(is_eos, as_tuple=False).view(-1):
                hyp = self._select(running_hyps, i_mask, b)
                ended_hyps[i_mask].append(hyp)

            remained_ids_list.append(
                torch.nonzero(is_eos == 0, as_tuple=False).view(-1).cpu()
            )

        running_hyps = self._batch_select_and_pad(running_hyps, remained_ids_list)
        return running_hyps

    def _get_new_mask_parallel_hyp(
        self,
        running_hyps: PartiallyARHypothesis,
        prev_hyp_ids: torch.Tensor,
        new_token_ids: torch.Tensor,
        weighted_scores: torch.Tensor,
        states: Dict[str, torch.Tensor],  # , part_states: Dict[str, torch.Tensor]
    ) -> PartiallyARHypothesis:
        # hyp.yseq.shape : (n_mask, n_beam, longest_yseq)
        n_mask = len(self.masks)
        new_yseq_length = running_hyps.yseq_length + 1
        yseq_shape = running_hyps.yseq.shape
        new_yseq = torch.zeros(
            (n_mask, self.beam_size, yseq_shape[-1] + 1),
            dtype=torch.int64,
            device=running_hyps.yseq.device,
        )
        new_yseq[:, :, :-1] = running_hyps.yseq[self.mask_ids, prev_hyp_ids]
        new_yseq[self.mask_ids, :, new_yseq_length.view(-1, 1) - 1] = (
            new_token_ids.unsqueeze(1)
        )  # (n_mask, 1, n_beam)

        current_score = weighted_scores.view(
            n_mask, self.beam_size, -1
        )  # (n_mask, n_beam, vocab_size)
        new_score = current_score[self.mask_ids, prev_hyp_ids, new_token_ids].view(
            -1
        )  # (n_mask, n_beam) => (n_mask * n_beam)

        new_states = dict()
        for k, v in states.items():
            # v is a torch.Tensor of shape (n_mask * n_beam, 1, D)
            # So we will select the previous states with slicing.
            # (n_mask * n_beam, layer, yseq_len, D)
            # => (n_mask, n_beam, layer, yseq_len, D)
            # => (n_mask, n_beam, layer, yseq_len, D)
            # => (n_mask * n_beam, layer, yseq_len, D)
            state_shape = v.size()
            new_states[k] = v.view(n_mask, self.beam_size, *state_shape[1:])[
                self.mask_ids, prev_hyp_ids
            ].reshape(n_mask * self.beam_size, *state_shape[1:])

        return PartiallyARHypothesis(
            score=new_score,
            yseq=new_yseq,
            yseq_length=new_yseq_length,
            states=new_states,
            eos=running_hyps.eos,
        )

    def _batch_select_and_pad(
        self, hyps: PartiallyARHypothesis, ids_list: List[torch.Tensor]
    ) -> PartiallyARHypothesis:
        # Since the number of remaining ids differs,
        # selecting and shifting the beam vector cannot be computed on parallel way
        # and we need to iterate over masks.
        n_mask = len(self.masks)
        new_yseq = torch.zeros(
            (n_mask, self.beam_size, hyps.yseq.shape[-1]),
            dtype=torch.int64,
            device=hyps.yseq.device,
        )
        new_score = torch.zeros(
            (n_mask, self.beam_size), dtype=hyps.score.dtype, device=hyps.score.device
        )
        for i_mask in range(n_mask):
            remaining_ids = ids_list[i_mask]
            self.num_hyps_for_masks[i_mask] = remaining_ids.size(0)

            new_yseq[i_mask, : remaining_ids.size(0)] = hyps.yseq[
                i_mask, remaining_ids
            ]  # (remain_beam, max_yseq_len)
            new_score[i_mask, : remaining_ids.size(0)] = hyps.score.view(n_mask, -1)[
                i_mask, remaining_ids
            ]

            for k, v in self.full_scorers.items():
                # v is a torch.Tensor of shape (n_mask * n_beam, layer, yseq_len, D)
                state_shape = hyps.states[k].size()
                current_states = hyps.states[k].view(
                    n_mask, self.beam_size, *state_shape[1:]
                )
                current_states[i_mask, : remaining_ids.size(0)] = current_states[
                    i_mask, remaining_ids
                ]
                hyps.states[k] = current_states.view(
                    n_mask * self.beam_size, *state_shape[1:]
                )

        return PartiallyARHypothesis(
            yseq=new_yseq,
            score=new_score.view(-1),
            yseq_length=hyps.yseq_length,
            states=hyps.states,
            eos=hyps.eos,
        )

    def _log_bests(self, bests: List[BatchHypothesis], maxlen, nbest_hyps_for_masks):
        for i, best in enumerate(bests):
            logging.info(f"total log prob (mask {i}): {best.score:.2f}")
            logging.info(
                f"normalized log prob (mask {i}): {best.score / len(best.yseq):.2f}"
            )
            logging.info(
                f"total number of ended hypo (mask {i}): {len(nbest_hyps_for_masks[i])}"
            )
            if self.token_list is not None:
                logging.info(
                    "best hypo: "
                    + "".join([self.token_list[x] for x in best.yseq[1:-1]])
                    + "\n"
                )
            if best.yseq[1:-1].shape[0] == maxlen:
                logging.warning(
                    "best hypo length: {} == max output length: {}".format(
                        best.yseq[1:-1].shape[0], maxlen
                    )
                )
                logging.warning(
                    "decoding may be stopped by the max output length limitation, "
                    + "please consider to increase the maxlenratio."
                )
