#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""GTN CTC implementation."""

import gtn
import torch


class GTNCTCLossFunction(torch.autograd.Function):
    """GTN CTC module."""

    # Copied from FB's GTN example implementation:
    # https://github.com/facebookresearch/gtn_applications/blob/master/utils.py#L251

    @staticmethod
    def create_ctc_graph(target, blank_idx):
        """Build gtn graph.

        :param list target: single target sequence
        :param int blank_idx: index of blank token
        :return: gtn graph of target sequence
        :rtype: gtn.Graph
        """
        g_criterion = gtn.Graph(False)
        L = len(target)
        S = 2 * L + 1
        for s in range(S):
            idx = (s - 1) // 2
            g_criterion.add_node(s == 0, s == S - 1 or s == S - 2)
            label = target[idx] if s % 2 else blank_idx
            g_criterion.add_arc(s, s, label)
            if s > 0:
                g_criterion.add_arc(s - 1, s, label)
            if s % 2 and s > 1 and label != target[idx - 1]:
                g_criterion.add_arc(s - 2, s, label)
        g_criterion.arc_sort(False)
        return g_criterion

    @staticmethod
    def forward(ctx, log_probs, targets, ilens, blank_idx=0, reduction="none"):
        """Forward computation.

        :param torch.tensor log_probs: batched log softmax probabilities (B, Tmax, oDim)
        :param list targets: batched target sequences, list of lists
        :param int blank_idx: index of blank token
        :return: ctc loss value
        :rtype: torch.Tensor
        """
        B, _, C = log_probs.shape
        losses = [None] * B
        scales = [None] * B
        emissions_graphs = [None] * B

        def process(b):
            # create emission graph
            T = ilens[b]
            g_emissions = gtn.linear_graph(T, C, log_probs.requires_grad)
            cpu_data = log_probs[b][:T].cpu().contiguous()
            g_emissions.set_weights(cpu_data.data_ptr())

            # create criterion graph
            g_criterion = GTNCTCLossFunction.create_ctc_graph(targets[b], blank_idx)
            # compose the graphs
            g_loss = gtn.negate(
                gtn.forward_score(gtn.intersect(g_emissions, g_criterion))
            )

            scale = 1.0
            if reduction == "mean":
                L = len(targets[b])
                scale = 1.0 / L if L > 0 else scale
            elif reduction != "none":
                raise ValueError("invalid value for reduction '" + str(reduction) + "'")

            # Save for backward:
            losses[b] = g_loss
            scales[b] = scale
            emissions_graphs[b] = g_emissions

        gtn.parallel_for(process, range(B))

        ctx.auxiliary_data = (losses, scales, emissions_graphs, log_probs.shape, ilens)
        loss = torch.tensor([losses[b].item() * scales[b] for b in range(B)])
        return torch.mean(loss.cuda() if log_probs.is_cuda else loss)

    @staticmethod
    def backward(ctx, grad_output):
        """Backward computation.

        :param torch.tensor grad_output: backward passed gradient value
        :return: cumulative gradient output
        :rtype: (torch.Tensor, None, None, None)
        """
        losses, scales, emissions_graphs, in_shape, ilens = ctx.auxiliary_data
        B, T, C = in_shape
        input_grad = torch.zeros((B, T, C))

        def process(b):
            T = ilens[b]
            gtn.backward(losses[b], False)
            emissions = emissions_graphs[b]
            grad = emissions.grad().weights_to_numpy()
            input_grad[b][:T] = torch.from_numpy(grad).view(1, T, C) * scales[b]

        gtn.parallel_for(process, range(B))

        if grad_output.is_cuda:
            input_grad = input_grad.cuda()
        input_grad *= grad_output / B

        return (
            input_grad,
            None,  # targets
            None,  # ilens
            None,  # blank_idx
            None,  # reduction
        )
