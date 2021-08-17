from typing import Dict
from typing import List

import k2
import torch


# Modified from: https://github.com/k2-fsa/icefall/blob/master/icefall/decode.py#L465
@torch.no_grad()
def rescore_with_whole_lattice(
    lats: k2.Fsa,
    G_with_epsilon_loops: k2.Fsa,
    lm_scale_list: List[float],
    need_rescored_lats: bool = False,
) -> Dict[str, k2.Fsa]:
    """Use whole lattice to rescore.

    Args:
      lats:
        An FsaVec It can be the output of `k2.intersect_dense_pruned`.
      G_with_epsilon_loops:
        An FsaVec representing the language model (LM). Note that it
        is an FsaVec, but it contains only one Fsa.
      lm_scale_list:
        A list containing lm_scale values.
    Returns:
      A dict of FsaVec, whose key is a lm_scale and the value represents the
      best decoding path for each sequence in the lattice.
    """
    assert len(lats.shape) == 3
    assert hasattr(lats, "lm_scores")
    assert G_with_epsilon_loops.shape == (1, None, None)

    device = lats.device
    lats.scores = lats.scores - lats.lm_scores
    # We will use lm_scores from G, so remove lats.lm_scores here
    del lats.lm_scores
    assert hasattr(lats, "lm_scores") is False

    # Now, lats.scores contains only am_scores

    # inverted_lats has word IDs as labels.
    # Its aux_labels are phone IDs, which is a ragged tensor k2.RaggedInt
    inverted_lats = k2.invert(lats)
    num_seqs = lats.shape[0]

    b_to_a_map = torch.zeros(num_seqs, device=device, dtype=torch.int32)
    try:
        rescoring_lats = k2.intersect_device(
            G_with_epsilon_loops, inverted_lats, b_to_a_map, sorted_match_a=True
        )
    except RuntimeError as e:
        print(f"Caught exception:\n{e}\n")
        print(f"Number of FSAs: {inverted_lats.shape[0]}")
        print("num_arcs before pruning: ", inverted_lats.arcs.num_elements())

        # NOTE(fangjun): The choice of the threshold 0.01 is arbitrary here
        # to avoid OOM. We may need to fine tune it.
        inverted_lats = k2.prune_on_arc_post(inverted_lats, 0.001, True)
        print("num_arcs after pruning: ", inverted_lats.arcs.num_elements())

        rescoring_lats = k2.intersect_device(
            G_with_epsilon_loops, inverted_lats, b_to_a_map, sorted_match_a=True
        )

    rescoring_lats = k2.top_sort(k2.connect(rescoring_lats))

    # inv_lats has phone IDs as labels
    # and word IDs as aux_labels.
    inv_lats = k2.invert(rescoring_lats)

    if need_rescored_lats:
        return inv_lats

    ans = dict()

    # The following implements
    # scores = (scores - lm_scores)/lm_scale + lm_scores
    #        = scores/lm_scale + lm_scores*(1 - 1/lm_scale)
    #
    saved_scores = inv_lats.scores.clone()
    for lm_scale in lm_scale_list:
        am_scores = saved_scores - inv_lats.lm_scores
        am_scores /= lm_scale
        inv_lats.scores = am_scores + inv_lats.lm_scores

        best_paths = k2.shortest_path(inv_lats, use_double_scores=True)
        key = f"lm_scale_{lm_scale}"
        ans[key] = best_paths
    return ans
