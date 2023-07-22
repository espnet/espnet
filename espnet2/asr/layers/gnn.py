from __future__ import division

import argparse
import json
import logging
import math
import os
import random
from copy import deepcopy
from time import time

import editdistance
import numpy as np
import six
import torch

from espnet.nets.pytorch_backend.nets_utils import to_device


class GCN(torch.nn.Module):
    def __init__(
        self,
        embdim: int,
        treehid: int,
        nlayer: int,
        dropout: float,
        residual: bool = False,
        tied: bool = False,
        nhead: int = 1,
        edgedrop: float = 0.0,
    ):
        super(GCN, self).__init__()
        self.treehid = treehid
        self.residual = residual
        self.embdim = embdim
        self.tied = tied
        self.nhead = nhead
        self.edgedrop = edgedrop
        if self.tied:
            assert treehid == embdim
        self.gcn_l1 = torch.nn.Linear(self.embdim, self.treehid)
        if self.residual:
            self.layernorm_l1 = torch.nn.LayerNorm(self.treehid)
        for i in range(nlayer - 1):
            setattr(
                self,
                "gcn_l{}".format(i + 2),
                torch.nn.Linear(self.treehid, self.treehid),
            )
            if self.tied and i < nlayer - 2:
                getattr(self, "gcn_l{}".format(i + 2)).weight = self.gcn_l1.weight
                getattr(self, "gcn_l{}".format(i + 2)).bias = self.gcn_l1.bias
            setattr(
                self, "layernorm_l{}".format(i + 2), torch.nn.LayerNorm(self.treehid)
            )
        self.dropout = torch.nn.Dropout(dropout)
        self.nlayer = nlayer

    def get_lextree_encs_gcn(self, lextree, embeddings, adjacency, wordpiece=None):
        if lextree[1] != -1 and wordpiece is not None:
            idx = len(embeddings)
            embeddings.append(wordpiece)
            adjacency.append([idx])
            lextree.append([])
            lextree.append(idx)
            return idx
        elif lextree[1] == -1 and lextree[0] != {}:
            ids = []
            idx = len(embeddings)
            if wordpiece is not None:
                embeddings.append(wordpiece)
            for newpiece, values in lextree[0].items():
                ids.append(
                    self.get_lextree_encs_gcn(values, embeddings, adjacency, newpiece)
                )
            if wordpiece is not None:
                adjacency.append([idx] + ids)
            lextree.append(ids)
            lextree.append(idx)
            return idx

    def forward_gcn(self, lextree, embeddings, adjacency, decemb):
        n_nodes = len(embeddings)
        nodes_encs = decemb.weight[embeddings]
        adjacency_mat = nodes_encs.new_zeros(n_nodes, n_nodes)
        for node in adjacency:
            for neighbour in node:
                adjacency_mat[node[0], neighbour] = 1.0
        degrees = torch.diag(torch.sum(adjacency_mat, dim=-1) ** -0.5)
        adjacency_mat = torch.einsum("ij,jk->ik", degrees, adjacency_mat)
        adjacency_mat = torch.einsum("ij,jk->ik", adjacency_mat, degrees)

        if self.training:
            edgedropmat = (
                torch.rand(adjacency_mat.size()).to(adjacency_mat.device)
                < self.edgedrop
            )
            adjacency_mat = adjacency_mat.masked_fill(edgedropmat, 0.0)
        all_node_encs = []
        for i in range(self.nlayer):
            next_nodes_encs = getattr(self, "gcn_l{}".format(i + 1))(nodes_encs)
            if i == 0:
                first_layer_enc = next_nodes_encs
            if self.nhead > 1 and i == 0:
                all_node_encs.append(next_nodes_encs)
            next_nodes_encs = torch.relu(
                torch.einsum("ij,jk->ik", adjacency_mat, next_nodes_encs)
            )
            if self.residual and i > 0:
                nodes_encs = next_nodes_encs + nodes_encs
                nodes_encs = getattr(self, "layernorm_l{}".format(i + 1))(nodes_encs)
            elif self.residual:
                nodes_encs = next_nodes_encs + first_layer_enc
                nodes_encs = getattr(self, "layernorm_l{}".format(i + 1))(nodes_encs)
            else:
                nodes_encs = next_nodes_encs
            all_node_encs.append(nodes_encs)
        if self.nhead > 1:
            output_encs = all_node_encs[0:1] + all_node_encs[-self.nhead + 1 :]
            nodes_encs = torch.cat(output_encs, dim=-1)
        return nodes_encs

    def fill_lextree_encs_gcn(self, lextree, nodes_encs, wordpiece=None):
        if lextree[1] != -1 and wordpiece is not None:
            idx = lextree[4]
            lextree[3] = nodes_encs[idx].unsqueeze(0)
        elif lextree[1] == -1 and lextree[0] != {}:
            idx = lextree[4]
            for newpiece, values in lextree[0].items():
                self.fill_lextree_encs_gcn(values, nodes_encs, newpiece)
            lextree[3] = nodes_encs[idx].unsqueeze(0)

    def forward(self, prefixtree, decemb):
        embeddings, adjacency = [], []
        self.get_lextree_encs_gcn(prefixtree, embeddings, adjacency)
        nodes_encs = self.forward_gcn(prefixtree, embeddings, adjacency, decemb)
        return nodes_encs
