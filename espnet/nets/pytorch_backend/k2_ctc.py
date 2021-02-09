#!/usr/bin/env python3
# Copyright (c)  2021  Mobvoi Inc. (authors: Yaguang Hu)

"""k2 based ctc loss implementation."""

from functools import lru_cache
import math
from typing import List
from typing import Tuple

import k2
import logging
import torch


class K2CTCLoss(torch.nn.Module):
    """K2CTCLoss module."""

    def __init__(
        self,
        odim: int,
        reduction: str = "sum",
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """K2CTCLoss module.

        Args:
          odim:
            Number of classes (including blank).
          reduction:
            Specifies the reduction to apply to the output. Default: 'sum'
          device:
            An instance of `torch.device`. Default: 'cpu'
        """
        torch.nn.Module.__init__(self)
        self.device = device
        self.reduction = reduction
        if reduction not in ["sum", "none"]:
            logging.error(
                f"k2 ctc loss reduction type: {reduction} is not supported yet,"
                "change it to be 'sum'"
            )
            self.reduction = "sum"
        self.graph_compiler = CtcTrainingGraphCompiler(odim, device=device)

    def forward(
        self,
        log_probs: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
    ) -> torch.Tensor:
        """k2 ctc forward function.

        Args:
          log_probs:
            Tensor of size (T, N, C), where T is input length, N is batch size,
            and C is number of classes (including blank). Note that it must be
            log probabilities.
          targets:
            Tensor of size (sum(target_lengths)). It represent corresponding labels
            of log_probs.
          input_lengths:
            Tuple or tensor of size (N). It represent the lengths of the inputs.
          target_lengths:
            Tuple or tensor of size (N). It represent lengths of the targets.
        """
        log_probs = log_probs.permute(
            1, 0, 2
        )  # now log_probs is [N, T, C]  batchSize x seqLength x alphabet_size
        supervision_segments = torch.stack(
            (
                torch.tensor(range(input_lengths.shape[0])),
                torch.zeros(input_lengths.shape[0]),
                input_lengths,
            ),
            1,
        ).to(torch.int32)

        dense_fsa_vec = k2.DenseFsaVec(log_probs, supervision_segments)

        decoding_graph = self.graph_compiler.compile(targets, target_lengths)
        if self.device != log_probs.device:
            decoding_graph = decoding_graph.to(log_probs.device)

        target_graph = k2.intersect_dense(decoding_graph, dense_fsa_vec, 100.0)
        tot_scores = target_graph.get_tot_scores(
            log_semiring=True, use_double_scores=True
        )
        if self.reduction == "none":
            tot_scores = tot_scores / target_lengths.to(log_probs.device)
        (tot_score, _, _) = get_tot_objf_and_num_frames(
            tot_scores, supervision_segments[:, 2]
        )
        return -tot_score


def build_ctc_topo(tokens: List[int]) -> k2.Fsa:
    """Build CTC topology.

    A token which appears once on the right side (i.e. olabels) may
    appear multiple times on the left side (ilabels), possibly with
    epsilons in between.

    When 0 appears on the left side, it represents the blank symbol;
    when it appears on the right side, it indicates an epsilon. That
    is, 0 has two meanings here.

    Args:
      tokens:
        A list of tokens, e.g., phones, characters, etc.
    Returns:
      Returns an FST that converts repeated tokens to a single token.
    """
    assert 0 in tokens, "We assume 0 is ID of the blank symbol"

    num_states = len(tokens)
    final_state = num_states
    arcs = ""
    for i in range(num_states):
        for j in range(num_states):
            if i == j:
                arcs += f"{i} {i} {tokens[i]} 0 0.0\n"
            else:
                arcs += f"{i} {j} {tokens[j]} {tokens[j]} 0.0\n"
        arcs += f"{i} {final_state} -1 -1 0.0\n"
    arcs += f"{final_state}"
    ans = k2.Fsa.from_str(arcs)
    ans = k2.arc_sort(ans)
    return ans


class CtcTrainingGraphCompiler(object):
    """CTC graph compiler."""

    def __init__(self, odim: int, device: torch.device):
        """CTC training graph compiler Module.

        Args:
          odim:
            Output dimension of CTC linear layer,
            len(symbol_list) + 2 (<blank> and <eos>).

        """
        self.dim = odim
        self.device = device
        self.ctc_topo_inv = k2.arc_sort(
            build_ctc_topo(list(range(self.dim))).invert_().to(self.device)
        )

    def compile(self, texts: torch.Tensor, texts_lengths: torch.Tensor) -> k2.Fsa:
        """CTC graph compilation function.

        Args:
          texts: Labels of log_probs.
          texts_lengths: Lengths of the targets.

        """
        texts_lengths = torch.cat([torch.tensor([0]), texts_lengths])
        texts_end_index = torch.cumsum(texts_lengths, 0)

        decoding_graphs = k2.create_fsa_vec(
            [
                self._compile_one_and_cache(
                    texts[texts_end_index[i] : texts_end_index[i + 1]]
                )
                for i in range(texts_lengths.shape[0] - 1)
            ]
        )

        # make sure the gradient is not accumulated
        decoding_graphs.requires_grad_(False)
        return decoding_graphs

    @lru_cache(maxsize=100000)
    def _compile_one_and_cache(self, text: torch.Tensor) -> k2.Fsa:
        label_graph = k2.linear_fsa(text.tolist(), self.device)
        label_graph = k2.add_epsilon_self_loops(label_graph)
        decoding_graph = k2.intersect(
            self.ctc_topo_inv, label_graph, treat_epsilons_specially=False
        )
        decoding_graph = k2.connect(decoding_graph.invert_())
        return decoding_graph


def get_tot_objf_and_num_frames(
    tot_scores: torch.Tensor, frames_per_seq: torch.Tensor
) -> Tuple[torch.Tensor, int, int]:
    """Filter all successful supervision segments.

    Args:
      tot_scores: A Torch tensor of shape (num_segments,) containing total scores
                 from forward-backward.
      frames_per_seq: A Torch tensor of shape (num_segments,) containing the
                      number of frames for each segment.

    Returns:
      Returns a tuple of 3 scalar tensors:  (tot_score, ok_frames, all_frames).
        where ok_frames is the frames for successful (finite) segments, and
        all_frames is the frames for all segments (finite or not).

    """
    mask = torch.ne(tot_scores, -math.inf)
    # finite_indexes is a tensor containing successful segment indexes, e.g.
    # [ 0 1 3 4 5 ]
    finite_indexes = torch.nonzero(mask).squeeze(1)
    # print("finite_indexes = ", finite_indexes, ", tot_scores = ", tot_scores)
    ok_frames = frames_per_seq[finite_indexes].sum()
    all_frames = frames_per_seq.sum()
    return (tot_scores[finite_indexes].sum(), ok_frames, all_frames)
