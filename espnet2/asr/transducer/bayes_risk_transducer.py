# Official Implementation of Bayes Risk Transducer:
# https://arxiv.org/pdf/2308.10107.pdf

import torch
import torch.nn.functional as F

try:
    import _k2  # k2 internal APIs
    import k2
except:
    raise ImportError("To use Bayes Risk Transducer, please install k2")

# A very large beam size ensures all lattice states / arcs are reserved
BEAM_SIZE = 1e10


class BayesRiskTransducer(torch.nn.Module):
    def __init__(
        self,
        risk_strategy,
        group_strategy,
        risk_factor=0.0,
        risk_start=0.0,
        pad_id=0,
    ):
        super().__init__()

        self.risk_strategy = risk_strategy
        self.group_strategy = group_strategy
        self.risk_factor = risk_factor
        self.risk_start = risk_start
        self.pad_id = pad_id

        self.rnnt_graph_buf = {}

    def forward(self, hs_pad, ys_pad, hlens, olens):
        # [B, T, U, D] -> [B, U, T, D]: be compatible with transducer interface
        hs_pad = hs_pad.transpose(1, 2)

        # (1) As required by k2, ordering by overall path lengths
        tot_lens = hlens + torch.ne(ys_pad, self.pad_id).int().sum(1)
        indices = torch.argsort(tot_lens, descending=True)
        hs_pad, ys_pad, hlens, olens = (
            hs_pad[indices],
            ys_pad[indices],
            hlens[indices],
            olens[indices],
        )
        nnet_output = F.log_softmax(hs_pad, dim=-1)

        # (2) compute the loss for each utterance
        loss = self.forward_core(nnet_output, ys_pad, hlens, olens)

        # (3) recover the original order
        indices2 = torch.argsort(indices)
        loss = loss[indices2]

        return loss.mean()

    def forward_core(self, nnet_output, ys_pad, hlens, olens):
        # (1) find size variables and validate them
        _olens = torch.ne(ys_pad, self.pad_id).int().sum(1)
        B, U_, T, D = nnet_output.size()
        U = max(olens)
        tot_lens = hlens + olens

        assert B == len(hlens) == len(ys_pad)
        assert U_ == U + 1
        assert T == max(hlens)
        assert torch.all(torch.eq(_olens, olens))

        # (2) build supervision and DenseFsaVec
        # Use this dummpy_nnet_output to build the lattice;
        # and then assign the real nnet_output to the lattice and compute the loss
        dummy_nnet_output = torch.zeros(
            (B, U + T, D), device=nnet_output.device, dtype=nnet_output.dtype
        )

        supervision = torch.stack(
            [torch.arange(B), torch.zeros(B), tot_lens.cpu()], dim=1
        ).int()

        dense_fsa_vec = k2.DenseFsaVec(dummy_nnet_output, supervision)

        # (3) Intersection to get lattice, which is for auto-grad
        ys = [[x for x in y if x != self.pad_id] for y in ys_pad.cpu().tolist()]
        graphs = self.compile_rnnt_graph(ys).to(nnet_output.device)
        lats = k2.intersect_dense(graphs, dense_fsa_vec, BEAM_SIZE)

        # (4) Find all forward-backward variables
        # Since the arc_map is not accessible to k2 users, do the intersect for the
        # second time with the interal _k2 APIs.
        with torch.no_grad():
            ragged_lat, arc_map_a, arc_map_b = _k2.intersect_dense(
                a_fsas=graphs.arcs,
                b_fsas=dense_fsa_vec.dense_fsa_vec,
                a_to_b_map=None,
                output_beam=BEAM_SIZE,
            )
            (
                (arc_u_idx, arc_t_idx, arc_k_idx, arc_b_idx),
                (state_u_idx, state_t_idx, state_k_idx, state_b_idx),
            ) = self.find_all_index(
                ragged_lat, graphs, dense_fsa_vec, arc_map_a, arc_map_b
            )

        # revise the lattice scores before forward-backward on the lattice
        zero_idx = (arc_k_idx == -1).nonzero(as_tuple=True)[0]
        arc_k_idx = torch.clip(arc_k_idx, min=0)
        arc_u_idx = arc_u_idx - 1
        rnnt_arc_u_idx = arc_u_idx // 2
        rnnt_arc_t_idx = torch.clip(arc_t_idx - rnnt_arc_u_idx, max=T - 1)

        # should compute the tmp_scores and assign to lats.scores with 2 lines
        # otherwise there is a bug due to k2 internal mechanism.
        tmp_scores = nnet_output[arc_b_idx, rnnt_arc_u_idx, rnnt_arc_t_idx, arc_k_idx]
        tmp_scores[zero_idx] = 0.0
        lats.scores = tmp_scores

        # From now on the arc_idx will not be used
        del arc_u_idx, arc_t_idx, arc_k_idx, arc_b_idx

        forward_scores = lats.get_forward_scores(True, True)
        backward_scores = lats.get_backward_scores(True, True)
        state_mask = torch.eq(state_k_idx, -1)  # k for super-nodes is -1
        forward_scores = torch.where(state_mask, float("-inf"), forward_scores)
        backward_scores = torch.where(state_mask, float("-inf"), backward_scores)

        state_t_idx = state_t_idx - state_u_idx // 2

        alpha = torch.ones([B, 2 * U + 3, T + 1]).double().to(
            nnet_output.device
        ) * float("-inf")
        beta = torch.ones([B, 2 * U + 3, T + 1]).double().to(
            nnet_output.device
        ) * float("-inf")

        alpha[state_b_idx, state_u_idx, state_t_idx] = forward_scores
        beta[state_b_idx, state_u_idx, state_t_idx] = backward_scores

        # remove the score of super-nodes
        alpha, beta = alpha[:, 1:-1], beta[:, 1:-1]

        # from now on the state_idx will not be used
        del state_u_idx, state_t_idx, state_k_idx, state_b_idx

        # (5) group strategy and risk strategy
        if self.group_strategy in ["transducer_ending", "transducer_ending_mean"]:
            loss_state = alpha + beta
            loss_state = loss_state[:, 1::2]  # [B, U, T + 1]

            loss_state = loss_state + self.get_risk_scores(loss_state, hlens, olens)
            loss_u = torch.logsumexp(loss_state, dim=2)
            mask = torch.ne(ys_pad, self.pad_id)

            if self.group_strategy == "transducer_ending_relative":
                loss_fsas = torch.where(mask, loss_u, 0.0).sum(1) / mask.double().sum(1)
            elif self.group_strategy == "transducer_ending":
                loss_fsas = loss_u[torch.arange(B).long(), mask.long().sum(1) - 1]
            else:
                raise NotImplementedError

        else:
            raise NotImplementedError

        # (6) avoid -inf loss. usually this means a bad case
        if torch.any(torch.isinf(loss_fsas)):
            loss_fsas = torch.where(torch.isinf(loss_fsas), 0.0, loss_fsas)

        # utterance-wise loss
        return -loss_fsas

    def get_risk_scores(self, loss_state, hlens, olens):
        """Add the bayes risk in multiple ways"""
        B, U, T = loss_state.size()

        if self.risk_strategy == "transducer_exp_relu":
            risk = (
                torch.arange(1, T + 1, device=loss_state.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(B, U, 1)
            )
            risk = torch.clip(risk - olens.view(B, 1, 1) * self.risk_start, min=0.0)
            risk = risk / hlens.unsqueeze(1).unsqueeze(1) * self.risk_factor

        elif self.risk_strategy == "transducer_exp_relative":
            risk = (
                torch.arange(0, T, device=loss_state.device)
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(B, U, 1)
            )
            max_stamp = torch.argmax(loss_state, dim=2, keepdim=True)
            risk[:, 1:] = torch.where(
                risk[:, 1:] <= max_stamp[:, :-1], 10000, risk[:, 1:]
            )
            risk = (
                (risk - max_stamp) / hlens.unsqueeze(1).unsqueeze(1) * self.risk_factor
            )

        else:
            raise NotImplementedError

        return -risk

    def find_all_index(
        self, ragged_lat, ctc_graph, dense_fsa_vec, arc_map_a, arc_map_b
    ):
        # This function finds the u_idx, t_idx, k_idx and b_idx for each lattice arc and each state
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

        # post-process: start and end states are the super-nodes. make the redundancy
        state_t_idx = state_t_idx + 1

        state_t_idx[start_state_id], state_u_idx[start_state_id] = 0, 0
        state_b_idx[start_state_id] = torch.arange(num_fsas, device=ctc_graph.device)

        return (arc_u_idx, arc_t_idx, arc_k_idx, arc_b_idx), (
            state_u_idx,
            state_t_idx,
            state_k_idx,
            state_b_idx,
        )

    def compile_rnnt_graph(self, ys):
        fsas = []
        for y in ys:
            if len(y) not in self.rnnt_graph_buf:
                self.rnnt_graph_buf[len(y)] = self.rnnt_graph_template(len(y))
            template = self.rnnt_graph_buf[len(y)]
            s = template.format(*y)  # fill the numbers
            fsa = k2.Fsa.from_str(s, num_aux_labels=1)
            fsa = k2.arc_sort(fsa)
            fsas.append(fsa)
        fsas = k2.create_fsa_vec(fsas)
        return fsas

    def rnnt_graph_template(self, n):
        s = ""
        for i in range(n + 1):
            s += f"{2*i} {2*i+1} 0 0 0.0\n"
            if i == n:
                pass  # must end with blank
            else:
                s += f"{2*i} {2*(i+1)} [] [] 0.0\n"
            s += f"{2*i+1} {2*i+1} 0 0 0.0\n"
            if i == n:
                s += f"{2*i+1} {2*(i+1)} -1 -1 0.0\n"
            else:
                s += f"{2*i+1} {2*(i+1)} [] [] 0.0\n"
        s += f"{2*n+2}"

        for i in range(n):
            sub = "{" + str(i) + "}"
            s = s.replace("[]", sub, 4)
        return s
