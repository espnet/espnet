# Official implementation of Bayes Risk CTC
# https://openreview.net/forum?id=Bd7GueaTxUz

import logging

import _k2  # k2 internal APIs
import k2
import torch


class BayesRiskCTC(torch.nn.Module):
    """
    Implements Bayes Risk CTC (Connectionist Temporal Classification).

    This class computes the Bayes Risk CTC loss, which incorporates
    risk strategies to improve the training of neural networks for
    sequence-to-sequence tasks, such as automatic speech recognition (ASR).

    Attributes:
        risk_strategy (str): Strategy for calculating risk. Options are
            "exp" for exponential risk or "exp_rel" for relative
            exponential risk.
        group_strategy (str): Strategy for grouping tokens. Options are
            "end" to consider the last token of each sequence or
            "end_mean" to average over the groups.
        risk_factor (float): A scaling factor for the risk value.

    Args:
        risk_strategy (str): Risk strategy to use. Defaults to "exp".
        group_strategy (str): Group strategy to use. Defaults to "end".
        risk_factor (float): Risk factor to apply. Defaults to 0.0.

    Returns:
        torch.Tensor: The computed Bayes Risk CTC loss for the input
        sequences.

    Raises:
        AssertionError: If an unknown risk_strategy or group_strategy
        is provided.
        NotImplementedError: If a user-defined group strategy is used
        that is not implemented.

    Examples:
        >>> model = BayesRiskCTC(risk_strategy="exp", group_strategy="end")
        >>> nnet_output = torch.randn(5, 10, 20)  # (B, T, C)
        >>> ys_pad = torch.randint(0, 20, (5, 10))  # (B, T)
        >>> hlens = torch.randint(1, 11, (5,))  # (B,)
        >>> ylens = torch.randint(1, 11, (5,))  # (B,)
        >>> loss = model(nnet_output, ys_pad, hlens, ylens)
        >>> print(loss)

    Note:
        The input `nnet_output` must be in the shape (B, T, C) where
        B is the batch size, T is the time dimension, and C is the
        number of classes.
    """

    def __init__(self, risk_strategy="exp", group_strategy="end", risk_factor=0.0):
        super().__init__()

        assert risk_strategy in ["exp", "exp_rel"], "Unknown risk_strategy for BRCTC"
        assert group_strategy in ["end", "end_mean"], "Unknown group_strategy for BRCTC"

        self.risk_strategy = risk_strategy
        self.group_strategy = group_strategy
        self.risk_factor = risk_factor

    def forward(self, nnet_output, ys_pad, hlens, ylens):
        """
        Computes the Bayes Risk CTC loss given neural network outputs and targets.

        This method processes the neural network outputs and calculates the
        Bayes Risk CTC loss. It first reorders and filters the inputs to ensure
        that the input lengths are in descending order, and then computes the
        loss using a core implementation method.

        Args:
            nnet_output (torch.Tensor): The output of the neural network with shape
                (B, T, C), where B is the batch size, T is the sequence length,
                and C is the number of classes.
            ys_pad (torch.Tensor): The padded target sequences with shape (B, max_len).
            hlens (torch.Tensor): The actual lengths of the input sequences with shape (B,).
            ylens (torch.Tensor): The actual lengths of the target sequences with shape (B,).

        Returns:
            torch.Tensor: A tensor containing the computed Bayes Risk CTC loss for each
            valid input in the batch.

        Raises:
            NotImplementedError: If a custom group strategy other than 'end' or 'end_mean'
            is used.

        Examples:
            >>> br_ctc = BayesRiskCTC(risk_strategy='exp', group_strategy='end')
            >>> nnet_output = torch.randn(3, 5, 10)  # Example output from NN
            >>> ys_pad = torch.tensor([[1, 2, 3], [1, 2, 0], [1, 0, 0]])
            >>> hlens = torch.tensor([5, 3, 1])
            >>> ylens = torch.tensor([3, 2, 1])
            >>> loss = br_ctc(nnet_output, ys_pad, hlens, ylens)
            >>> print(loss)

        Note:
            Ensure that the lengths of the sequences are correctly managed before
            calling this method to avoid invalid inputs.
        """
        # Reorder and filter out invalid examples:
        # A. K2 requires that hlens are in descending order;
        # B. remove all examples whose hlens is smaller than necessary.
        indices = torch.argsort(hlens, descending=True)
        ys, min_hlens = self.find_minimum_hlens(ys_pad[indices], ylens[indices])
        valid_sample_indices = (min_hlens <= hlens[indices]).nonzero(as_tuple=True)[0]

        if len(valid_sample_indices) < 1:
            logging.warning(
                "All examples are invalid for Bayes Risk CTC. Skip this batch"
            )
            return torch.Tensor([0.0]).to(nnet_output.device)

        indices = indices[valid_sample_indices]
        nnet_output, hlens, ylens = nnet_output[indices], hlens[indices], ylens[indices]
        ys = [ys[i.item()] for i in valid_sample_indices]

        # Core implementation
        loss_utt = self.forward_core(nnet_output, ys, hlens, ylens)

        # Recover the original order. Invalid examples are excluded.
        indices2 = torch.argsort(indices)
        loss_utt = loss_utt[indices2]

        return loss_utt

    def forward_core(self, nnet_output, ys, hlens, ylens):
        """
        Compute the core forward pass for the Bayes Risk CTC.

        This method calculates the loss for the given neural network outputs
        using the Bayes Risk CTC framework. It involves building a dense
        FSA from the network outputs and intersecting it with CTC graphs to
        derive the forward and backward scores.

        Args:
            nnet_output (torch.Tensor): The neural network output of shape
                (B, T, C) where B is the batch size, T is the time steps,
                and C is the number of classes.
            ys (list): A list of target sequences (padded) for the batch.
            hlens (torch.Tensor): A tensor of shape (B,) containing the lengths
                of each sequence in the batch.
            ylens (torch.Tensor): A tensor of shape (B,) containing the lengths
                of each target sequence.

        Returns:
            torch.Tensor: A tensor containing the computed loss for each
            example in the batch.

        Raises:
            NotImplementedError: If a custom group strategy is used that
            is not implemented.

        Examples:
            >>> nnet_output = torch.randn(3, 10, 5)  # Batch of 3, 10 time steps, 5 classes
            >>> ys = [[1, 2, 3], [2, 3, 4], [1, 1, 2]]
            >>> hlens = torch.tensor([10, 10, 10])
            >>> ylens = torch.tensor([3, 3, 3])
            >>> br_ctc = BayesRiskCTC()
            >>> loss = br_ctc.forward_core(nnet_output, ys, hlens, ylens)
            >>> print(loss)

        Note:
            Ensure that the input tensors are on the same device as the
            neural network output for correct computations.
        """
        # (1) Find the shape
        (B, T, _), U = nnet_output.size(), max(ylens)

        # (2) Build DenseFsaVec and CTC graphs
        supervision = torch.stack(
            [torch.arange(B), torch.zeros(B), hlens.cpu()], dim=1
        ).int()

        dense_fsa_vec = k2.DenseFsaVec(nnet_output, supervision)
        ctc_graphs = k2.ctc_graph(ys).to(nnet_output.device)

        # (3) Intersection for twice
        #     One is for back-prop, the other is to find arc_maps
        #     A. In K2, the arc_maps are not provided in user API.
        #     This would slightly slow down the training but can
        #     ensure that code is compatible with original K2 API
        #     B. Always use very large beam size to avoid any pruning
        #     so the results are identical to vanilla CTC if no
        #     risk values are applied.
        lats = k2.intersect_dense(ctc_graphs, dense_fsa_vec, 1e20)

        with torch.no_grad():
            ragged_lat, arc_map_a, arc_map_b = _k2.intersect_dense(
                a_fsas=ctc_graphs.arcs,
                b_fsas=dense_fsa_vec.dense_fsa_vec,
                a_to_b_map=None,
                output_beam=1e20,
            )

            (arc_u_idx, arc_t_idx, arc_k_idx, arc_b_idx), (
                state_u_idx,
                state_t_idx,
                state_k_idx,
                state_b_idx,
            ) = self.find_all_index(
                ragged_lat, ctc_graphs, dense_fsa_vec, arc_map_a, arc_map_b
            )

        # (4) Rearrange all forward-backward variables and emission probabilities
        # into matrices. Always use double precision to reduce accumulated error.
        forward_scores = lats.get_forward_scores(True, True)
        backward_scores = lats.get_backward_scores(True, True)

        alpha = torch.ones([B, 2 * U + 3, T + 2]).double().to(
            nnet_output.device
        ) * float("-inf")
        beta = torch.ones([B, 2 * U + 3, T + 2]).double().to(
            nnet_output.device
        ) * float("-inf")

        alpha[state_b_idx, state_u_idx, state_t_idx] = forward_scores
        beta[state_b_idx, state_u_idx, state_t_idx] = backward_scores
        alpha, beta = alpha[:, 1:-1, 1:-1], beta[:, 1:-1, 1:-1]

        p = torch.ones([B, 2 * U + 1, T]).double().to(nnet_output.device) * float(
            "-inf"
        )
        state_u_idx = torch.clip(state_u_idx - 1, min=0, max=2 * U)
        state_t_idx = torch.clip(state_t_idx - 1, min=0, max=T - 1)
        p[state_b_idx, state_u_idx, state_t_idx] = nnet_output.double()[
            state_b_idx, state_t_idx, state_k_idx
        ]
        p = torch.where(torch.isinf(alpha), float("-inf"), p)

        # (5) Compute the summed posterior within each group;
        #     apply the risk value and compute the overall loss
        if self.group_strategy in ["end", "end_mean"]:
            # Only consider non-blank tokens
            alpha = alpha[:, 1::2]
            beta = beta[:, 1::2]
            p = p[:, 1::2]

            beta_prime = log_substraction_exp(
                beta[:, :, :-1], beta[:, :, 1:] + p[:, :, 1:]
            )
            beta_prime = torch.cat([beta_prime, beta[:, :, -1:]], dim=-1)
            loss_state = alpha + beta_prime

            # Apply risk values
            loss_state = loss_state + self.get_risk_scores(
                loss_state, hlens, self.risk_factor
            )

            # aggregate all groups
            loss_u = torch.logsumexp(loss_state, dim=2)
            mask = ~torch.isinf(loss_u)

            if self.group_strategy == "end_mean":
                loss_fsas = torch.where(mask, loss_u, 0.0).sum(1) / mask.double().sum(1)

            else:
                loss_fsas = loss_u[torch.arange(B).long(), mask.long().sum(1) - 1]

            return -loss_fsas

        # Users may design their own group strategy here
        else:
            raise NotImplementedError

    def get_risk_scores(self, loss_state, hlens, risk_factor):
        """
        Calculate Bayes risk scores based on the provided strategy.

        This method computes the risk scores that can be added to the loss
        states in order to adjust the training process according to the
        defined risk strategy. It supports two strategies: "exp" and
        "exp_rel". The computed risk scores are negative values to be
        consistent with the loss minimization objective.

        Attributes:
            loss_state (torch.Tensor): A tensor of shape (B, U, T) representing
                the loss states for each batch, utterance, and time step.
            hlens (torch.Tensor): A tensor of shape (B,) representing the
                lengths of each input sequence in the batch.
            risk_factor (float): A scalar factor that scales the risk values.

        Args:
            loss_state (torch.Tensor): A tensor containing loss states with
                dimensions (B, U, T).
            hlens (torch.Tensor): A tensor containing the lengths of the
                sequences with shape (B,).
            risk_factor (float): A scaling factor for the risk computation.

        Returns:
            torch.Tensor: A tensor containing the calculated risk scores of
            shape (B, U, T), which are negative values to be added to the
            loss.

        Raises:
            NotImplementedError: If an unknown risk strategy is specified.

        Examples:
            >>> loss_state = torch.randn(2, 3, 5)  # Example loss states
            >>> hlens = torch.tensor([5, 4])  # Example lengths
            >>> risk_factor = 0.1
            >>> risk_scores = get_risk_scores(loss_state, hlens, risk_factor)
            >>> print(risk_scores.shape)
            torch.Size([2, 3, 5])

        Note:
            The implementation uses broadcasting to compute the risk scores
            based on the specified strategy and the lengths of the sequences.
        """
        B, U, T = loss_state.size()

        if self.risk_strategy == "exp":
            risk = (
                torch.arange(1, T + 1, device=loss_state.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(B, U, 1)
            )
            risk = risk / hlens.unsqueeze(1).unsqueeze(1) * risk_factor

        elif self.risk_strategy == "exp_rel":
            risk = (
                torch.arange(1, T + 1, device=loss_state.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(B, U, 1)
            )
            max_stamp = torch.argmax(loss_state, dim=2, keepdim=True)
            risk = (risk - max_stamp) / hlens.unsqueeze(1).unsqueeze(1) * risk_factor

        else:
            raise NotImplementedError

        return -risk

    def find_all_index(
        self, ragged_lat, ctc_graph, dense_fsa_vec, arc_map_a, arc_map_b
    ):
        """
        Finds the indices of (b, t, u, d) for each arc and state.

        This function processes the input data structures to extract
        relevant indices that correspond to the arcs in the CTC graph
        and the states in the dense FSA vector. It performs several
        computations to ensure that the indices are accurately mapped
        from the ragged lattice.

        Args:
            ragged_lat (k2.Fsa): The ragged lattice representing the
                lattice structure.
            ctc_graph (k2.Fsa): The CTC graph containing the arcs
                and their structure.
            dense_fsa_vec (k2.DenseFsaVec): The dense FSA vector
                that holds the neural network output.
            arc_map_a (torch.Tensor): A mapping tensor for arcs in
                the CTC graph.
            arc_map_b (torch.Tensor): A mapping tensor for arcs in
                the dense FSA vector.

        Returns:
            tuple: A tuple containing two elements:
                - (torch.Tensor, torch.Tensor, torch.Tensor,
                  torch.Tensor): The arc indices corresponding to
                  the CTC graph arcs.
                - (torch.Tensor, torch.Tensor, torch.Tensor,
                  torch.Tensor): The state indices corresponding
                  to the states in the CTC graph.

        Examples:
            Given a ragged lattice and corresponding CTC graph and
            dense FSA vector, the function can be used as follows:

            >>> arc_indices, state_indices = find_all_index(
            ...     ragged_lat, ctc_graph, dense_fsa_vec, arc_map_a, arc_map_b
            ... )

        Note:
            This function is intended for internal use within the
            BayesRiskCTC implementation and should not be called
            directly by users.

        Todo:
            Implement additional error handling for invalid input
            types or shapes.
        """
        # This function finds the index of (b, t, u, d) for each arc and each state
        num_fsas = len(ctc_graph.arcs.row_splits(1)) - 1

        # Arc part
        ctc_graph_arcs = ctc_graph.as_dict()["arcs"][2 * num_fsas + 4 :].view(-1, 4)
        arc_u_idx = ctc_graph_arcs[:, 1].long()
        arc_u_idx = arc_u_idx[arc_map_a.long()]

        arc_k_idx = ctc_graph_arcs[:, 2].long()
        arc_k_idx = arc_k_idx[arc_map_a.long()]

        arc_boundaries = (
            ctc_graph.arcs.shape()
            .row_splits(2)
            .long()[ctc_graph.arcs.shape().row_splits(1).long()]
        )
        arc_ids = torch.arange(ctc_graph.num_arcs, device=ctc_graph.device)
        arc_b_idx = torch.bucketize(arc_ids, arc_boundaries, right=True) - 1
        arc_b_idx = arc_b_idx[arc_map_a.long()]

        arc_t_idx = arc_map_b // dense_fsa_vec.scores.size(1)
        duration = dense_fsa_vec.duration + 1
        t_shift = torch.zeros(
            num_fsas + 1, dtype=duration.dtype, device=ctc_graph.device
        )
        t_shift[1:] = torch.cumsum(duration, dim=0)
        arc_t_idx = (arc_t_idx - t_shift[arc_b_idx]).long()

        # State part
        lats = k2.Fsa(ragged_lat)
        arc2state = lats._get_entering_arcs(
            False
        ).long()  # encoming arc_id for start state is -1
        start_state_id = lats.arcs.row_splits(1)[:-1].long()

        state_u_idx, state_t_idx, state_k_idx, state_b_idx = (
            arc_u_idx[arc2state],
            arc_t_idx[arc2state],
            arc_k_idx[arc2state],
            arc_b_idx[arc2state],
        )

        # Post-process:
        # A. Start and end states are the super-nodes. make the redundancy.
        # B. Correct the values on start states.
        # C. Clip k_idx to avoid any negative value.
        state_u_idx, state_t_idx = state_u_idx + 1, state_t_idx + 1

        state_t_idx[start_state_id], state_u_idx[start_state_id] = 0, 0
        state_b_idx[start_state_id] = torch.arange(num_fsas, device=ctc_graph.device)
        state_k_idx, arc_k_idx = torch.clip(state_k_idx, min=0), torch.clip(
            arc_k_idx, min=0
        )

        return (arc_u_idx, arc_t_idx, arc_k_idx, arc_b_idx), (
            state_u_idx,
            state_t_idx,
            state_k_idx,
            state_b_idx,
        )

    def find_minimum_hlens(self, ys_pad, ylens):
        """
        Finds the minimum lengths of the hypotheses and their padded forms.

        This function processes the padded sequences of hypotheses and their
        corresponding lengths to determine the minimum length required for each
        hypothesis. The minimum length is calculated by considering consecutive
        tokens and counting repetitions. The function returns a list of the
        processed hypotheses and a tensor containing the minimum lengths.

        Args:
            ys_pad (torch.Tensor): A tensor of shape (B, T) where B is the batch
                size and T is the maximum sequence length. Each row contains the
                padded hypothesis sequences.
            ylens (torch.Tensor): A tensor of shape (B,) containing the actual
                lengths of each hypothesis sequence.

        Returns:
            tuple: A tuple containing:
                - list: A list of processed hypotheses where each hypothesis is
                  represented as a list of integers.
                - torch.Tensor: A tensor of shape (B,) containing the minimum
                  lengths for each hypothesis.

        Examples:
            >>> ys_pad = torch.tensor([[1, 2, 2, 0], [1, 1, 2, 3]])
            >>> ylens = torch.tensor([3, 4])
            >>> ys, min_hlens = find_minimum_hlens(ys_pad, ylens)
            >>> print(ys)
            [[1, 2, 2], [1, 1, 2, 3]]
            >>> print(min_hlens)
            tensor([3, 4])

        Note:
            The minimum length is calculated by counting each token and
            considering repetitions. For example, if a token repeats, it adds
            to the minimum length.
        """
        device = ys_pad.device
        ys_pad, ylens = ys_pad.cpu().tolist(), ylens.cpu().tolist()
        ys, min_hlens = [], []

        for y_pad, ylen in zip(ys_pad, ylens):
            y, min_hlen = [], 0
            prev = None

            for i in range(ylen):
                y.append(y_pad[i])
                min_hlen += 1

                if y_pad[i] == prev:
                    min_hlen += 1

                prev = y_pad[i]

            ys.append(y)
            min_hlens.append(min_hlen)

        min_hlens = torch.Tensor(min_hlens).long().to(device)

        return ys, min_hlens


def log_substraction_exp(a, b):
    """
    Numerically stable computation of log(a.exp() - b.exp()).

    This function computes the logarithm of the difference between
    the exponentials of two tensors in a numerically stable way to
    prevent overflow or underflow issues that can occur when directly
    calculating the exponentials of large or small values.

    The computation is performed as follows:
        log_substraction_exp(a, b) = log(exp(a) - exp(b))

    This implementation ensures that operations on -inf values are handled
    appropriately, avoiding invalid computations.

    Args:
        a (torch.Tensor): The first input tensor.
        b (torch.Tensor): The second input tensor.

    Returns:
        torch.Tensor: A tensor containing the result of log(exp(a) - exp(b)).

    Examples:
        >>> a = torch.tensor([1.0, 2.0, float('-inf')])
        >>> b = torch.tensor([0.5, 1.5, float('-inf')])
        >>> log_substraction_exp(a, b)
        tensor([0.3133, 0.3133, -inf])

    Note:
        This function assumes that the input tensors are of the same shape.
        If they are not, broadcasting rules will apply.
    """
    ans = torch.ones_like(a) * float("-inf")

    # avoid -inf in input
    mask1 = torch.logical_and(~torch.isinf(a), ~torch.isinf(b))
    a_ = torch.where(mask1, a, -1.0)  # avoid any operation on -inf
    b_ = torch.where(mask1, b, -2.0)

    # avoid -inf in output: need to be very small
    # as these values would be picked by mask1
    ans_tmp = b_ + ((a_ - b_).exp() - 1).log()
    a_ = torch.where(torch.isinf(ans_tmp), -2000.0, a_)
    b_ = torch.where(torch.isinf(ans_tmp), -2001.0, b_)

    ans1 = b_ + ((a_ - b_).exp() - 1).log()
    ans = torch.where(mask1, ans1, ans)

    mask2 = torch.logical_and(~torch.isinf(a), torch.isinf(b))
    ans = torch.where(mask2, a, ans)

    return ans
