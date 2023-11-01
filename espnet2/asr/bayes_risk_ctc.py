# Official implementation of Bayes Risk CTC
# https://openreview.net/forum?id=Bd7GueaTxUz

import logging

import _k2  # k2 internal APIs
import k2
import torch


class BayesRiskCTC(torch.nn.Module):
    def __init__(self, risk_strategy="exp", group_strategy="end", risk_factor=0.0):
        super().__init__()

        assert risk_strategy in ["exp", "exp_rel"], "Unknown risk_strategy for BRCTC"
        assert group_strategy in ["end", "end_mean"], "Unknown group_strategy for BRCTC"

        self.risk_strategy = risk_strategy
        self.group_strategy = group_strategy
        self.risk_factor = risk_factor

    def forward(self, nnet_output, ys_pad, hlens, ylens):
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
        """Add the bayes risk in multiple ways"""
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
    """return (a.exp() - b.exp()).log(): a numeracal safe implementation"""
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
