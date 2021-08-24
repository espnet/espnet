import k2
import math
import torch

from typing import Dict
from typing import List
from typing import Tuple


def remove_repeated_and_leq(tokens: List[int], blank_id: int = 0):
    """Generate valid token sequence.

    Result may be used as input of transformer decoder and neural language model.
    Fristly, remove repeated token from a "token alignment" seqs;
    Then remove blank symbols.
    This fuction may be replaced by tokenizing word_seqs with tokenizer
    or composeing word_seqs_fsas with L_inv.fst
    or composing token_seqs with ctc_topo.
    Current method is slelected other than previous three methods
    because it won't need an extra object, i.e. tokenizer, L.fst or ctc_topo.
    """
    new_tokens = []
    previous = None
    for token in tokens:
        if token != previous:
            new_tokens.append(token)
            previous = token
    new_tokens = [token for token in new_tokens if token > blank_id]
    return new_tokens


def _intersect_device(
    a_fsas: k2.Fsa, b_fsas: k2.Fsa, b_to_a_map: torch.Tensor, sorted_match_a: bool
):
    """Wrap k2.intersect_device.

    its purpose is to split b_fsas into several batches
    and process each batch separately to avoid CUDA OOM error.
    The arguments and return value of this function are the same as
    k2.intersect_device.
    """
    # NOTE: You can decrease batch_size in case of CUDA out of memory error.
    batch_size = 500
    num_fsas = b_fsas.shape[0]
    if num_fsas <= batch_size:
        return k2.intersect_device(
            a_fsas, b_fsas, b_to_a_map=b_to_a_map, sorted_match_a=sorted_match_a
        )

    num_batches = int(math.ceil(float(num_fsas) / batch_size))
    splits = []
    for i in range(num_batches):
        start = i * batch_size
        end = min(start + batch_size, num_fsas)
        splits.append((start, end))

    ans = []
    for start, end in splits:
        indexes = torch.arange(start, end).to(b_to_a_map)

        fsas = k2.index(b_fsas, indexes)
        b_to_a = k2.index(b_to_a_map, indexes)
        path_lats = k2.intersect_device(
            a_fsas, fsas, b_to_a_map=b_to_a, sorted_match_a=sorted_match_a
        )
        ans.append(path_lats)

    return k2.cat(ans)


def compute_am_scores_and_lm_scores(
    lats: k2.Fsa, word_fsas_with_epsilon_loops: k2.Fsa, path_to_seq_map: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute AM and LM scores of n-best lists (represented as word_fsas).

    Args:
      lats:
        An FsaVec, which is the output of `k2.intersect_dense_pruned`.
        It must have the attribute `lm_scores`.
      word_fsas_with_epsilon_loops:
        An FsaVec representing a n-best list. Note that it has been processed
        by `k2.add_epsilon_self_loops`.
      path_to_seq_map:
        A 1-D torch.Tensor with dtype torch.int32. path_to_seq_map[i] indicates
        which sequence the i-th Fsa in word_fsas_with_epsilon_loops belongs to.
        path_to_seq_map.numel() == word_fsas_with_epsilon_loops.arcs.dim0().
    Returns:
      Return a tuple of (1-D torch.Tensor, 1-D torch.Tensor)
      containing the AM and FM scores of each path.
      `am_scores.numel() == word_fsas_with_epsilon_loops.shape[0]`
      `lm_scores.numel() == word_fsas_with_epsilon_loops.shape[0]`
    """
    assert len(lats.shape) == 3

    # k2.compose() currently does not support b_to_a_map. To void
    # replicating `lats`, we use k2.intersect_device here.
    #
    # lats has phone IDs as `labels` and word IDs as aux_labels, so we
    # need to invert it here.
    inverted_lats = k2.invert(lats)

    # Now the `labels` of inverted_lats are word IDs (a 1-D torch.Tensor)
    # and its `aux_labels` are phone IDs ( a k2.RaggedInt with 2 axes)

    # Remove its `aux_labels` since it is not needed in the
    # following computation
    del inverted_lats.aux_labels
    inverted_lats = k2.arc_sort(inverted_lats)

    am_path_lats = _intersect_device(
        inverted_lats,
        word_fsas_with_epsilon_loops,
        b_to_a_map=path_to_seq_map,
        sorted_match_a=True,
    )

    am_path_lats = k2.top_sort(k2.connect(am_path_lats))

    # The `scores` of every arc consists of `am_scores` and `lm_scores`
    if hasattr(lats, "lm_scores"):
        am_path_lats.scores = am_path_lats.scores - am_path_lats.lm_scores
        am_scores = am_path_lats.get_tot_scores(
            use_double_scores=True, log_semiring=False
        )

        # Start to compute lm_scores
        am_path_lats.scores = am_path_lats.lm_scores
        lm_scores = am_path_lats.get_tot_scores(
            use_double_scores=True, log_semiring=False
        )
    else:
        am_scores = am_path_lats.get_tot_scores(
            use_double_scores=True, log_semiring=False
        )
        lm_scores = None

    return am_scores, lm_scores


def nbest_am_lm_scores(lats: k2.Fsa, num_paths: int):
    """Compute am scores and lm scores with word_seqs.

    Compatible with both ctc_decoding or TLG decoding.
    """
    paths = k2.random_paths(lats, num_paths=num_paths, use_double_scores=True)
    if hasattr(lats.aux_labels, "contiguous"):
        word_seqs = k2.index(lats.aux_labels.contiguous(), paths)
    else:
        # '_k2.RaggedInt' object has no attribute 'contiguous'
        word_seqs = k2.index(lats.aux_labels, paths)

    # With ctc_decoding, word_seqs stores token_ids.
    # With TLG decoding, word_seqs stores word_ids.
    word_seqs = k2.ragged.remove_values_leq(word_seqs, 0)
    unique_word_seqs, num_repeats, new2old = k2.ragged.unique_sequences(
        word_seqs, need_num_repeats=True, need_new2old_indexes=True
    )

    seq_to_path_shape = k2.ragged.get_layer(unique_word_seqs.shape(), 0)
    path_to_seq_map = seq_to_path_shape.row_ids(1)
    # used to split final computed tot_scores
    seq_to_path_splits = seq_to_path_shape.row_splits(1)

    unique_word_seqs = k2.ragged.remove_axis(unique_word_seqs, 0)
    word_fsas = k2.linear_fsa(unique_word_seqs)

    word_fsas_with_epsilon_loops = k2.add_epsilon_self_loops(word_fsas)

    am_scores, lm_scores = compute_am_scores_and_lm_scores(
        lats, word_fsas_with_epsilon_loops, path_to_seq_map
    )

    token_seqs = k2.index(lats.labels.contiguous(), paths)
    token_seqs = k2.ragged.remove_axis(token_seqs, 0)
    token_ids, _ = k2.ragged.index(token_seqs, new2old, axis=0)
    token_ids = k2.ragged.to_list(token_ids)
    # Now remove repeated tokens and 0s and -1s.
    token_ids = [remove_repeated_and_leq(tokens) for tokens in token_ids]
    return am_scores, lm_scores, token_ids, new2old, path_to_seq_map, seq_to_path_splits


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
        saved_scores = inv_lats.scores.clone()
        am_scores = saved_scores - inv_lats.lm_scores
        inv_lats.scores = am_scores + 0.5 * inv_lats.lm_scores
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
