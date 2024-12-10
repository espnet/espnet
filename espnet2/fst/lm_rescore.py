import math
from typing import List, Tuple

import torch

try:
    import k2
except (ImportError, ModuleNotFoundError):
    k2 = None


def remove_repeated_and_leq(tokens: List[int], blank_id: int = 0):
    """
    Generate a valid token sequence by removing repetitions and blanks.

    This function processes a sequence of tokens by first removing 
    consecutive repeated tokens, then filtering out any tokens that 
    are less than or equal to the specified blank ID. The resulting 
    sequence can be utilized as input for a transformer decoder or 
    a neural language model.

    This method is chosen over alternatives such as tokenizing word 
    sequences with a tokenizer, composing word sequences with 
    `L_inv.fst`, or using a CTC topology, as it does not require 
    additional objects like a tokenizer or finite state transducer 
    (FST).

    Args:
        tokens (List[int]): A list of token integers to process.
        blank_id (int, optional): The threshold for blank tokens. Tokens 
            less than or equal to this value will be removed. Defaults 
            to 0.

    Returns:
        List[int]: A new list of tokens with repetitions and blanks 
        removed.

    Examples:
        >>> remove_repeated_and_leq([1, 2, 2, 3, 0, 4, 0])
        [1, 2, 3, 4]

        >>> remove_repeated_and_leq([0, 1, 1, 0, 2, 3], blank_id=1)
        [2, 3]
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
    """
    Compute Acoustic Model (AM) and Language Model (LM) scores for n-best lists.

    This function computes the AM and LM scores for a given n-best list of
    hypotheses represented as finite state acceptors (FSAs). The input FSAs
    are expected to have been processed appropriately and must adhere to
    specific formats required by the underlying k2 library.

    Args:
      lats: 
        An FsaVec output from `k2.intersect_dense_pruned`. It must contain 
        the attribute `lm_scores`.
      word_fsas_with_epsilon_loops: 
        An FsaVec representing the n-best list, which has been processed by 
        `k2.add_epsilon_self_loops`.
      path_to_seq_map: 
        A 1-D tensor of type `torch.int32`. Each entry `i` indicates which 
        sequence the i-th Fsa in `word_fsas_with_epsilon_loops` belongs to. 
        The size of this tensor must match the number of arcs in 
        `word_fsas_with_epsilon_loops`.
      device: 
        A string specifying the device to perform computations on (default: 
        "cuda"). It can be set to "cpu" or "cuda" based on the available 
        hardware.
      batch_size: 
        An integer that defines the batch size for processing the n-best 
        list when intersecting with `lats`. This can be adjusted to avoid 
        GPU out-of-memory errors.

    Returns:
      A tuple containing two 1-D tensors: the first tensor represents the 
      AM scores, and the second tensor represents the LM scores for each 
      path in the n-best list. Both tensors will have a size equal to the 
      number of FSAs in `word_fsas_with_epsilon_loops`.

    Raises:
      AssertionError: If the `k2` module is not installed or if the 
      input `lats` does not have the correct shape.

    Examples:
      >>> lats = k2.Fsa(...)
      >>> word_fsas = k2.Fsa(...)
      >>> path_to_seq_map = torch.tensor([...], dtype=torch.int32)
      >>> am_scores, lm_scores = compute_am_scores_and_lm_scores(
      ...     lats, word_fsas, path_to_seq_map
      ... )

    Note:
      Ensure that the k2 library is properly installed and configured before 
      using this function.
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
    """
    Compute acoustic model (AM) scores and language model (LM) scores for 
    n-best paths from the provided lattice.

    This function computes the AM scores and LM scores based on the 
    n-best paths generated from the input lattice. It is compatible with 
    both CTC decoding and TLG decoding methods.

    Args:
        lats (k2.Fsa): An FSA (finite-state acceptor) representing the 
            lattice containing the paths from which scores will be computed.
        num_paths (int): The number of n-best paths to compute from the 
            input lattice.
        device (str, optional): The device to perform computations on 
            (default is "cuda"). It can be set to "cpu" for CPU 
            computations.
        batch_size (int, optional): The size of batches to process during 
            the computation to manage memory usage (default is 500).

    Returns:
        Tuple[torch.Tensor, torch.Tensor, List[List[int]], 
        torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - am_scores: A 1-D tensor of acoustic model scores for each path.
            - lm_scores: A 1-D tensor of language model scores for each path 
              (may be None if not applicable).
            - token_ids: A list of lists containing token IDs for each path 
              after removing repeated tokens and blank tokens.
            - new2old: A tensor mapping new token IDs to their original IDs.
            - path_to_seq_map: A tensor mapping paths to sequences.
            - seq_to_path_splits: A tensor representing the splits of 
              sequences to paths.

    Raises:
        AssertionError: If `k2` is not installed or if the input lattice is 
            not valid.

    Examples:
        >>> am_scores, lm_scores, token_ids, new2old, path_to_seq_map, \
        ... seq_to_path_splits = nbest_am_lm_scores(lats, num_paths=10)
        >>> print(am_scores.shape)  # Output: torch.Size([10])
        >>> print(len(token_ids))    # Output: 10

    Note:
        The `k2` library must be installed for this function to work. 
        Follow the installation instructions in 'tools/installers'.
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
