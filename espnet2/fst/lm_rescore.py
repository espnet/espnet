import math
from typing import List, Tuple

import torch

try:
    import k2  # for CI import
except ImportError or ModuleNotFoundError:
    k2 = None


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
    a_fsas: k2.Fsa,
    b_fsas: k2.Fsa,
    b_to_a_map: torch.Tensor,
    sorted_match_a: bool,
    batch_size: int = 500,
):
    """Wrap k2.intersect_device

    This is a wrapper of k2.intersect_device and its purpose is to split
    b_fsas into several batches and process each batch separately to avoid
    CUDA OOM error.
    The arguments and return value of this function are the same as
    k2.intersect_device.

    NOTE: You can decrease batch_size in case of CUDA out of memory error.
    """
    assert k2 is not None, "please follow 'tools/installers' to install"

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

        fsas = k2.index_fsa(b_fsas, indexes)
        b_to_a = k2.index_select(b_to_a_map, indexes)
        path_lats = k2.intersect_device(
            a_fsas, fsas, b_to_a_map=b_to_a, sorted_match_a=sorted_match_a
        )
        ans.append(path_lats)

    return k2.cat(ans)


def compute_am_scores_and_lm_scores(
    lats: k2.Fsa,
    word_fsas_with_epsilon_loops: k2.Fsa,
    path_to_seq_map: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 500,
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
      batch_size:
        Batchify the n-best list when intersecting with inverted_lats.
        You could tune this to avoid GPU OOM issue or increase the GPU usage.
    Returns:
      Return a tuple of (1-D torch.Tensor, 1-D torch.Tensor) containing
      the AM and LM scores of each path.
      `am_scores.numel() == word_fsas_with_epsilon_loops.shape[0]`
      `lm_scores.numel() == word_fsas_with_epsilon_loops.shape[0]`
    """
    assert (
        k2 is not None
    ), "k2 is not installed, please follow 'tools/installers' to install"
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
        batch_size=batch_size,
    )

    am_path_lats = k2.top_sort(k2.connect(am_path_lats))

    # The `scores` of every arc consists of `am_scores` and `lm_scores`
    tot_score_device = "cpu"
    if hasattr(lats, "lm_scores"):
        am_path_lats.scores = am_path_lats.scores - am_path_lats.lm_scores
        am_scores = (
            am_path_lats.to(tot_score_device)
            .get_tot_scores(use_double_scores=True, log_semiring=False)
            .to(device)
        )

        # Start to compute lm_scores
        am_path_lats.scores = am_path_lats.lm_scores
        lm_scores = (
            am_path_lats.to(tot_score_device)
            .get_tot_scores(use_double_scores=True, log_semiring=False)
            .to(device)
        )
    else:
        am_scores = (
            am_path_lats.to(tot_score_device)
            .get_tot_scores(use_double_scores=True, log_semiring=False)
            .to(device)
        )
        lm_scores = None

    return am_scores, lm_scores


def nbest_am_lm_scores(
    lats: k2.Fsa,
    num_paths: int,
    device: str = "cuda",
    batch_size: int = 500,
):
    """Compute am scores with word_seqs

    Compatible with both ctc_decoding or TLG decoding.
    """
    assert (
        k2 is not None
    ), "k2 is not installed, please follow 'tools/installers' to install"
    paths = k2.random_paths(lats, num_paths=num_paths, use_double_scores=True)
    if isinstance(lats.aux_labels, torch.Tensor):
        word_seqs = k2.ragged.index(lats.aux_labels.contiguous(), paths)
    else:
        # '_k2.RaggedInt' object has no attribute 'contiguous'
        word_seqs = lats.aux_labels.index(paths)
        word_seqs = word_seqs.remove_axis(word_seqs.num_axes - 2)

    # With ctc_decoding, word_seqs stores token_ids.
    # With TLG decoding, word_seqs stores word_ids.
    word_seqs = word_seqs.remove_values_leq(0)
    unique_word_seqs, num_repeats, new2old = word_seqs.unique(
        need_num_repeats=True, need_new2old_indexes=True
    )

    seq_to_path_shape = unique_word_seqs.shape.get_layer(0)
    path_to_seq_map = seq_to_path_shape.row_ids(1)
    # used to split final computed tot_scores
    seq_to_path_splits = seq_to_path_shape.row_splits(1)

    unique_word_seqs = unique_word_seqs.remove_axis(0)
    word_fsas = k2.linear_fsa(unique_word_seqs)

    word_fsas_with_epsilon_loops = k2.add_epsilon_self_loops(word_fsas)

    am_scores, lm_scores = compute_am_scores_and_lm_scores(
        lats, word_fsas_with_epsilon_loops, path_to_seq_map, device, batch_size
    )

    token_seqs = k2.ragged.index(lats.labels.contiguous(), paths)
    token_seqs = token_seqs.remove_axis(0)

    token_ids, _ = token_seqs.index(new2old, axis=0)
    token_ids = token_ids.tolist()
    # Now remove repeated tokens and 0s and -1s.
    token_ids = [remove_repeated_and_leq(tokens) for tokens in token_ids]
    return am_scores, lm_scores, token_ids, new2old, path_to_seq_map, seq_to_path_splits
